# synthesizer_cli.py
import argparse
import numpy as np
import torch
import sys # <-- Add sys import

# --- START OF FIX: Make the script aware of the custom packages path ---
persistent_packages_path = '/storage/my_packages'
if persistent_packages_path not in sys.path:
    sys.path.append(persistent_packages_path)
# --- END OF FIX ---

# Now the rest of the imports will work
from utils import load_preset
from synthesis import synthesize_fm_chain
from synthesis_torch import synthesize_fm_chain_torch
from generate_wave_file import save_wave_file

def render_command(args):
    """The logic for the 'render' sub-command."""
    try:
        params, metadata = load_preset(args.preset_path)
    except FileNotFoundError:
        print(f"Error: Preset file not found at '{args.preset_path}'")
        return

    synthesis_method = metadata.get('optimization_method')
    print(f"Synthesizing a {args.duration}-second version using the '{synthesis_method}' engine...")

    if synthesis_method == 'adam':
        params_tensor = torch.from_numpy(params).float()
        signal_tensor = synthesize_fm_chain_torch(params_tensor, args.duration, args.sr)
        final_signal = signal_tensor.detach().numpy()
    elif synthesis_method in ['differential_evolution', 'dual_annealing', 'cma']:
        final_signal = synthesize_fm_chain(params, args.duration, args.sr)
    else:
        print(f"Error: Unknown synthesis method '{synthesis_method}' found in preset.")
        return

    if args.output:
        output_filename = args.output
    else:
        preset_name = args.preset_path.split('/')[-1].replace('.json', '')
        output_filename = f"{preset_name}_{args.duration}s.wav"

    save_wave_file(
        signal=final_signal,
        source_sr=args.sr,
        target_sr=args.sr,
        filename=output_filename
    )

def main():
    # --- Main parser setup ---
    parser = argparse.ArgumentParser(description="A command-line tool for the FM Synthesizer.")
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # --- 'render' command setup ---
    render_parser = subparsers.add_parser("render", help="Render audio from a saved preset.")
    render_parser.add_argument("preset_path", type=str, help="Path to the JSON preset file.")
    render_parser.add_argument("--duration", type=float, default=5.0, help="Duration of the output audio in seconds.")
    render_parser.add_argument("--output", type=str, default=None, help="Name of the output WAV file.")
    render_parser.add_argument("--sr", type=int, default=44100, help="Sample rate of the output audio.")
    # Link this parser to the render_command function
    render_parser.set_defaults(func=render_command)
    
    # You could add other commands here in the future, e.g., 'analyze' or 'optimize'
    # analyze_parser = subparsers.add_parser("analyze", help="Analyze an audio file.")
    # analyze_parser.set_defaults(func=analyze_command)

    # Parse the arguments and call the appropriate function
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()