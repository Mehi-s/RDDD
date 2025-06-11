# predict.py
#
# Performs inference using a trained model on an input image and saves the output.
# The script loads a pre-trained network, processes an input image,
# runs the image through the model, and saves the resulting image.
#
# Example Usage:
# python predict.py --model_path pretrained/cls_model.pth --image_path path/to/your/image.jpg --output_path path/to/output/image.png --gpu_ids 0
#
# Ensure that 'pretrained/cls_model.pth' (or your specified model) exists and is accessible.
# The 'data' and 'models' directories from the project also need to be accessible for imports.

import argparse
import torch
# PIL (Pillow) for image loading and manipulation
from PIL import Image # Moved here as it's a primary dependency

def parse_args():
  """Parses command-line arguments for the prediction script.

  Defines arguments for model path, image path, output path, GPU IDs, and loss column.
  It also handles parsing and validation of GPU IDs.

  Returns:
    argparse.Namespace: An object containing the parsed command-line arguments.
  """
    parser = argparse.ArgumentParser(description="Image-to-Image Translation Prediction Script")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained model checkpoint file (.pth). This is typically for the classifier network (net_c).')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image file (e.g., .jpg, .png).')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the processed output image.')
    parser.add_argument('--gpu_ids', type=str, default='-1',
                        help="Comma-separated list of GPU IDs to use (e.g., '0', '0,1'). Use '-1' for CPU mode.")
    parser.add_argument('--loss_col', type=int, default=4,
                        help='Controls which output from the multi-scale network to use. Corresponds to a specific level of detail/scale in the model architecture.')
    args = parser.parse_args()

    # Parse and validate gpu_ids
    str_ids = args.gpu_ids.split(',')
    parsed_gpu_ids = []
    for str_id in str_ids:
        try:
            id_val = int(str_id)
            if id_val >= -1: # Allow -1 for CPU
                parsed_gpu_ids.append(id_val)
            else:
                print(f"Warning: Invalid GPU ID '{str_id}' found, ignoring.")
        except ValueError:
            print(f"Warning: Non-integer GPU ID '{str_id}' found, ignoring.")

    args.gpu_ids = [] # Will store actual CUDA device IDs or be empty for CPU
    if not parsed_gpu_ids or parsed_gpu_ids[0] == -1: # If CPU is chosen or no IDs provided
        print("Running on CPU.")
    elif torch.cuda.is_available():
        # Filter for valid and available GPU IDs
        for id_val in parsed_gpu_ids:
            if id_val >= 0 and id_val < torch.cuda.device_count():
                args.gpu_ids.append(id_val)
            elif id_val != -1: # Don't warn if -1 was part of a list like "0,-1"
                 print(f"Warning: GPU ID {id_val} is not available or invalid, ignoring.")

        if not args.gpu_ids: # If all specified GPUs were invalid but CUDA is available
             print("No valid GPUs specified or all specified GPUs are unavailable. Falling back to CPU.")
        else:
             print(f"Using GPU(s): {args.gpu_ids}")
             torch.cuda.set_device(args.gpu_ids[0]) # Set default CUDA device to the first valid one
    else: # CUDA not available, but GPU IDs were specified
        print("CUDA is not available, but GPU IDs were specified. Running on CPU.")

    return args

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    print("Parsed arguments:", args)

    # Determine computation device (CPU or GPU)
    # If specific, valid GPU IDs are provided and CUDA is available, use the first one. Otherwise, use CPU.
    if args.gpu_ids and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Import Project-Specific Modules ---
    # These are imported here to ensure paths are correctly handled,
    # assuming the script is run from a context where these modules are accessible.
    from data.transforms import to_norm_tensor # Image transformation utility
    from models import make_model # Factory function to create the model instance
    # Utility to convert tensor to a displayable NumPy image array
    from models.cls_model_eval_nocls_reg import tensor2im
    import numpy as np # For numerical operations, mainly for image conversion

    processed_image_tensor = None # Initialize to ensure it's defined in all execution paths

    # --- 1. Load and Preprocess Input Image ---
    print(f"Loading and preprocessing image: {args.image_path}")
    try:
        # Open the image file and convert to RGB format
        image = Image.open(args.image_path).convert('RGB')

        # Apply project-specific normalization and convert to a PyTorch tensor
        processed_image_tensor = to_norm_tensor(image)
        # Add a batch dimension (B, C, H, W) as models typically expect batches
        processed_image_tensor = processed_image_tensor.unsqueeze(0)
        # Move the tensor to the selected computation device (GPU or CPU)
        processed_image_tensor = processed_image_tensor.to(device)

        print(f"Processed image shape: {processed_image_tensor.shape}, Device: {processed_image_tensor.device}")
    except FileNotFoundError:
        print(f"Error: Input image file not found at {args.image_path}")
        exit(1) # Exit if the image file cannot be found
    except Exception as e:
        print(f"An error occurred during image loading or preprocessing: {e}")
        exit(1) # Exit on other image processing related errors

    # --- 2. Create Model Configuration (Options Namespace) ---
    # This 'Options' class mimics the configuration object ('opt') used during the model's training.
    # It provides necessary parameters for model initialization and structure.
    # Many of these are defaults from training scripts and may not all be strictly necessary for inference,
    # but are included for compatibility with the model's initialization code.
    class Options:
      """
      A configuration class that mimics the 'opt' namespace typically used during
      training. This allows the prediction script to initialize models in a way
      that is consistent with how they were trained, even if only a subset of
      these options are strictly required for inference.
      """
        def __init__(self):
            # --- Essential settings for prediction ---
            self.gpu_ids = args.gpu_ids         # Parsed list of integer GPU IDs (empty for CPU)
            self.isTrain = False              # Critical: ensures model is in evaluation mode
            self.model_path = args.model_path   # Path to the pre-trained net_c (classifier) model
            self.loss_col = args.loss_col     # Selects which output from the multi-scale generator to use

            # --- General model and network architecture parameters ---
            self.model = 'ytmtnet'            # Model type identifier (ClsModel is a specialized ytmtnet)
            self.netG = 'RDnet_'              # Name of the generator network architecture
            self.num_subnet = 4               # Number of sub-networks in RDnet_ (architecture specific)
            self.hyper = False                # Whether to use VGG hypercolumn features (typically False for this model)
            self.input_nc = 3                 # Number of input image channels (e.g., 3 for RGB)
            self.output_nc = 3                # Number of output image channels (e.g., 3 for RGB)
            self.ngf = 64                     # Base number of filters in the generator network
            self.norm = 'instance'            # Type of normalization layer (e.g., 'instance', 'batch')
            self.init_type = 'normal'         # Strategy for initializing network weights
            self.init_gain = 0.02             # Gain factor for some weight initialization strategies
            self.dropout_rate = 0             # Dropout rate (RDnet_ often uses drop_path instead)
            self.no_antialias = False         # Whether to disable antialiasing (advanced option)
            self.no_antialias_up = False      # Whether to disable antialiasing for upsampling (advanced option)

            # --- Settings primarily for training, provided for compatibility ---
            self.checkpoints_dir = './checkpoints' # Default directory for saving checkpoints (not used for loading one file)
            self.name = 'predict'             # Name for this prediction run (used if logging were active)
            self.serial_batches = True        # Process images one by one (standard for prediction)
            self.nThreads = 1                 # Number of threads for data loading (1 is fine for single image prediction)
            self.max_dataset_size = float('inf') # Maximum number of images to process in a dataset
            self.display_winsize = 256        # Window size for Visdom display (training utility)
            self.display_port = 8097          # Port for Visdom server
            self.display_id = 0               # Visdom display ID
            self.display_single_pane_ncols = 0 # Visdom display option
            self.no_log = True                # Disable detailed logging utilities
            self.no_verbose = True            # Disable verbose output from model components
            self.resume = False               # Do not attempt to resume training
            self.resume_epoch = None          # Epoch to resume from (if resume were True)
            self.seed = 2018                  # Random seed for reproducibility
            self.supp_eval = False            # Supplementary evaluation flag
            self.start_now = False            # Training utility flag
            self.testr = False                # Testing related flag
            self.select = None                # Selection criteria for data
            self.dataroot = './datasets/dummy' # Dummy path for dataset root (not used for single image)
            self.dataset_mode = 'single'      # Indicates single image inference rather than a dataset
            self.direction = 'AtoB'           # Data transformation direction (e.g., A->B)
            self.ndf = 64                     # Base number of filters in discriminator (not used in inference)
            # Loss function weights (not used in inference but might be checked by model init)
            self.lambda_GAN = 0
            self.lambda_idt = 0
            self.lambda_cycle = 0
            self.lambda_vgg = 0

    opt = Options()

    # --- 3. Model Loading and Inference ---
    if processed_image_tensor is not None: # Proceed only if image preprocessing was successful
        try:
            print("Initializing model...")
            # Create the model instance (ClsModel in this case, via the make_model factory)
            # The 'None' argument to make_model might be specific to this project's factory.
            # However, the make_model function in models/__init__.py does not use its argument.
            model = make_model(opt) # Pass opt, though it's not used by current make_model
            # Initialize the model with the configuration object.
            # This step typically loads network structures and may load some weights if defined in initialize.
            print("Initializing model with options...") # Added print statement for clarity
            model.initialize(opt)
            print("Model initialized successfully.")

            # Set the model to evaluation mode. This is crucial for consistent results during inference
            # as it disables layers like Dropout and uses running averages for BatchNorm.
            model.eval()
            print("Model set to evaluation mode.")

            print("Performing inference...")
            # The model's internal device handling (from initialize and .to(device) calls)
            # should manage tensor placement. processed_image_tensor is already on the correct device.

            # Forward pass through the classifier network (net_c)
            # This network might provide guidance or features to the main generator.
            ipt = model.net_c(processed_image_tensor)

            # Forward pass through the main image translation network (net_i or similar)
            # `prompt=True` might be a specific flag for this model's generator.
            _output_i, output_j_list = model.net_i(processed_image_tensor, ipt, prompt=True)

            # Select the desired output tensor based on the 'loss_col' argument.
            # 'output_j_list' typically contains multiple output tensors from different
            # stages or scales of the generator network (net_i).
            # In ClsModel's forward_eval, output_j is populated with [out_clean, out_reflection] pairs.
            # So, the index for the Nth 'out_clean' (where N is loss_col) is 2 * (N-1).
            output_index = 2 * (opt.loss_col - 1)

            predicted_tensor = None # Initialize to handle out-of-range cases
            if 0 <= output_index < len(output_j_list):
                predicted_tensor = output_j_list[output_index]
                print(f"Selected output from net_i stage corresponding to loss_col={opt.loss_col}. Predicted_tensor shape: {predicted_tensor.shape}")
            else:
                print(f"Error: loss_col {opt.loss_col} results in an invalid index {output_index} for the generator's output list (which has {len(output_j_list)} raw items).")
                print(f"Available 'clean image' outputs correspond to loss_col values from 1 to {len(output_j_list) // 2}.")

            # --- 4. Postprocess and Save Output Image ---
            if predicted_tensor is not None: # Proceed only if a valid tensor was selected
                print("Postprocessing and saving output image...")
                try:
                    # Convert the output tensor to a NumPy array suitable for image display/saving.
                    # tensor2im handles normalization and channel reordering if necessary.
                    numpy_image = tensor2im(predicted_tensor)

                    # Create a PIL (Pillow) Image from the NumPy array.
                    # Ensure data type is uint8 for standard image formats.
                    pil_image = Image.fromarray(numpy_image.astype(np.uint8))

                    # Save the processed image to the path specified by the user.
                    pil_image.save(args.output_path)
                    print(f"Output image successfully saved to: {args.output_path}")

                except Exception as e:
                    print(f"Error during output image processing or saving: {e}")

        except FileNotFoundError as e: # Specifically catch FileNotFoundError if model files are missing
             print(f"Error: A model file was not found. Please check paths. Details: {e}")
        except RuntimeError as e: # Catch other runtime errors that can occur during model operations
            # Common PyTorch errors include state_dict mismatches or CUDA errors.
            if "state_dict" in str(e) or "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                 print(f"Error during model loading (likely a state_dict mismatch or incorrect model file): {e}")
            elif "CUDA" in str(e):
                 print(f"CUDA-related error during model operation: {e}")
            else:
                print(f"An error occurred during model loading or inference: {e}")
        except Exception as e: # Catch any other unexpected errors
            print(f"An unexpected error occurred: {e}")
