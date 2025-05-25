# backend/app/services/llm_service.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from app.core.config import settings
from app.models.llm import RecommendationRequest, MealPlanRequest
from app.models.user import UserProfile

# --- Environment Setup (from app1.py) ---
# These might be less critical in a backend service but kept for consistency
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception as e:
    print(f"Warning: Could not set torch multiprocessing strategy: {e}")
    pass

# --- Global Variables for Model and Tokenizer ---
llm_model = None
llm_tokenizer = None
model_loading_error = None

def load_llm_model_and_tokenizer():
    """Loads the LLM model and tokenizer based on settings."""
    global llm_model, llm_tokenizer, model_loading_error
    if llm_model and llm_tokenizer:
        return # Already loaded

    try:
        print(f"Loading LLM model from: {settings.LLM_MODEL_ID}")
        print(f"Loading Tokenizer from: {settings.BASE_MODEL_ID}")

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # Use float32 for CPU
            bnb_4bit_use_double_quant=True,
        )

        # Load the model
        # Note: Quantization primarily benefits CUDA. CPU loading might be slow/inefficient.
        # Consider using a non-quantized model or different library (like llama.cpp) for CPU inference.
        model_kwargs = {
            "quantization_config": bnb_config if device == "cuda" else None,
            "device_map": "auto" if device == "cuda" else {"": "cpu"}, # Explicitly map to CPU if no CUDA
            "trust_remote_code": True,
            # Add torch_dtype if not using quantization on CPU
            # "torch_dtype": torch.float32 if device == "cpu" else None
        }
        # Remove quantization_config if on CPU, as BitsAndBytes might require CUDA
        if device == "cpu":
            del model_kwargs["quantization_config"]
            print("Warning: Running LLM on CPU without quantization. Performance may be limited.")

        llm_model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_ID,
            **model_kwargs
        )

        # Load the tokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_ID, trust_remote_code=True)
        # Ensure pad token is set if missing
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token")

        print("LLM Model and Tokenizer loaded successfully.")
        model_loading_error = None

    except Exception as e:
        model_loading_error = f"Failed to load LLM model or tokenizer: {e}. Check model paths and dependencies (CUDA drivers if using GPU)."
        print(f"ERROR: {model_loading_error}")
        llm_model = None
        llm_tokenizer = None

# --- Load model on application startup ---
# This will run when the module is first imported by FastAPI
load_llm_model_and_tokenizer()

# --- Service Functions ---

async def generate_recommendation(request: RecommendationRequest) -> str:
    """Generates a health recommendation based on food info and user profile."""
    if model_loading_error:
        return f"Error: LLM not available. {model_loading_error}"
    if not llm_model or not llm_tokenizer:
        return "Error: LLM model or tokenizer not loaded."
    if not request.food_info:
        return "Error: Food information is required to generate a recommendation."

    # Format user profile for the prompt
    profile_str = f"Age: {request.user_profile.age}, Height: {request.user_profile.height}cm, Weight: {request.user_profile.weight}kg, BMI: {request.user_profile.bmi:.1f}, Gender: {request.user_profile.gender}, Activity: {request.user_profile.activity_level}, Conditions: {request.user_profile.health_conditions}"

    system_message = {
        "role": "system",
        "content": f"You are a helpful nutrition assistant. Analyze the provided food information in the context of the user\'s profile and health condition. Food Information: {request.food_info}\nUser Profile: {profile_str}"
    }
    user_message = {
        "role": "user",
        "content": f"Based on the food info and my profile, is this food generally suitable considering I have {request.health_condition}? Provide a concise, high-level recommendation and general considerations."
    }
    messages = [system_message, user_message]

    try:
        # Ensure tokenizer has pad_token_id
        pad_token_id = llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id

        input_ids = llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(llm_model.device)

        # Generate response
        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=pad_token_id
            )

        # Decode the response
        response_ids = outputs[0][input_ids.shape[-1]:]
        recommendation = llm_tokenizer.decode(response_ids, skip_special_tokens=True)
        return recommendation.strip()

    except Exception as e:
        print(f"Error during LLM recommendation generation: {e}")
        return "Error generating recommendation."

async def generate_meal_plan(request: MealPlanRequest) -> str:
    """Generates a one-day meal plan based on user profile and health condition."""
    if model_loading_error:
        return f"Error: LLM not available. {model_loading_error}"
    if not llm_model or not llm_tokenizer:
        return "Error: LLM model or tokenizer not loaded."

    # Format user profile for the prompt
    profile_str = f"Age: {request.user_profile.age}, Height: {request.user_profile.height}cm, Weight: {request.user_profile.weight}kg, BMI: {request.user_profile.bmi:.1f}, Gender: {request.user_profile.gender}, Activity: {request.user_profile.activity_level}, Conditions: {request.user_profile.health_conditions}"

    system_message = {
        "role": "system",
        "content": f"You are a helpful meal planner. Create a suitable one-day meal plan based on the user\'s profile and health condition. User Profile: {profile_str}\nHealth Condition: {request.health_condition}"
    }
    user_message = {
        "role": "user",
        "content": "Generate a one-day meal plan (breakfast, lunch, dinner, and two snacks) suitable for my health condition and profile. Keep it concise and practical."
    }
    messages = [system_message, user_message]

    try:
        # Ensure tokenizer has pad_token_id
        pad_token_id = llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id

        input_ids = llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(llm_model.device)

        # Generate response
        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids,
                max_new_tokens=500, # Increased token limit for meal plan
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=pad_token_id
            )

        # Decode the response
        response_ids = outputs[0][input_ids.shape[-1]:]
        meal_plan = llm_tokenizer.decode(response_ids, skip_special_tokens=True)
        return meal_plan.strip()

    except Exception as e:
        print(f"Error during LLM meal plan generation: {e}")
        return "Error generating meal plan."

