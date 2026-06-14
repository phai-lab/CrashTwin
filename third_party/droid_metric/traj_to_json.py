import json
import os
import argparse

def read_transform_matrix(filepath):
    """Read transformation matrix from text file"""
    with open(filepath, 'r') as f:
        content = f.read().strip()

    # Parse the matrix values
    lines = content.split('\n')
    matrix = []

    for line in lines:
        if line.strip():
            # Remove line numbers and arrows
            if '→' in line:
                values_str = line.split('→')[1].strip()
            else:
                # Handle lines that might not have arrows
                values_str = line.strip()
                # Remove any leading numbers followed by tab
                if '\t' in values_str:
                    values_str = values_str.split('\t')[-1]

            # Parse the numerical values
            try:
                values = [float(v) for v in values_str.split()]
                if values:  # Only add non-empty rows
                    matrix.append(values)
            except:
                continue

    # Return 4x4 matrix
    return matrix[:4] if len(matrix) >= 4 else matrix

def convert_all_trajectories(input_dir, output_file):
    """Convert all trajectory files to a single JSON"""
    trajectories = {}

    # Get all txt files sorted
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    for filename in files:
        filepath = os.path.join(input_dir, filename)
        frame_id = filename.replace('.txt', '')

        try:
            matrix = read_transform_matrix(filepath)
            if matrix:
                trajectories[frame_id] = matrix
                print(f"Processed {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(trajectories, f, indent=2)

    print(f"\nSuccessfully converted {len(trajectories)} trajectory frames to {output_file}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert trajectory files to JSON')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing trajectory .txt files')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file path')
    args = parser.parse_args()

    convert_all_trajectories(args.input_dir, args.output_file)