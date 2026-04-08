import re
from typing import List, Tuple, Dict
from pathlib import Path

# ============================================================================
# CONFIGURATION - Set the path to your tracking annotation file here
# ============================================================================
PATH_annotation = r"/home/ucl/ingi/trixen/ChimpRec/ChimpVideos/20241018 - 07h56.txt"  # Change this to your file path
# Examples:
# PATH_annotation = r"./tracking_data/20241019 - 13h28.txt"
# PATH_annotation = r"/path/to/annotations/tracking_file.txt"
# ============================================================================


def validate_tracking_file(filepath: str) -> Dict:
    """
    Validates a chimpanzee tracking annotation file.
    
    Expected format:
    ChimpName: ID*start_frame-end_frame
    
    Example:
    Djiku: 4965*0-556
    Malago: 4986*888-1027
    Ivan: 4993*1136-1140
    
    Args:
        filepath: Path to the annotation file
        
    Returns:
        Dictionary with validation results including errors found
    """
    
    # Pattern: Name: Number*Number-Number
    pattern = r'^([A-Za-z]+):\s+(\d+)\*(\d+)-(\d+)\s*$'
    
    errors = []
    valid_lines = 0
    total_lines = 0
    chimp_frames = {}  # Track frame ranges per chimp
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, start=1):
            # Skip empty lines
            if line.strip() == '':
                continue
            
            total_lines += 1
            match = re.match(pattern, line.strip())
            
            if not match:
                errors.append({
                    'line': line_num,
                    'type': 'FORMAT_ERROR',
                    'message': f"Invalid format",
                    'content': line.strip(),
                    'expected': "ChimpName: ID*start_frame-end_frame"
                })
                continue
            
            chimp_name, chimp_id, start_frame, end_frame = match.groups()
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            
            # Check if start_frame <= end_frame
            if start_frame > end_frame:
                errors.append({
                    'line': line_num,
                    'chimp': chimp_name,
                    'type': 'FRAME_RANGE_ERROR',
                    'message': f"Start frame ({start_frame}) is greater than end frame ({end_frame})",
                    'content': line.strip()
                })
                continue
            
            # Check for overlapping frames for the same chimp
            if chimp_name not in chimp_frames:
                chimp_frames[chimp_name] = []
            
            # Check overlap with existing ranges
            for existing_start, existing_end, existing_line in chimp_frames[chimp_name]:
                if not (end_frame < existing_start or start_frame > existing_end):
                    errors.append({
                        'line': line_num,
                        'chimp': chimp_name,
                        'type': 'OVERLAP_WARNING',
                        'message': f"Frame range {start_frame}-{end_frame} overlaps with line {existing_line} ({existing_start}-{existing_end})",
                        'content': line.strip()
                    })
            
            chimp_frames[chimp_name].append((start_frame, end_frame, line_num))
            valid_lines += 1
        
        return {
            'status': 'VALID' if len(errors) == 0 else 'INVALID',
            'filepath': filepath,
            'total_lines': total_lines,
            'valid_lines': valid_lines,
            'errors': errors,
            'chimp_summary': {name: len(ranges) for name, ranges in chimp_frames.items()}
        }
    
    except FileNotFoundError:
        return {
            'status': 'ERROR',
            'filepath': filepath,
            'message': f"File not found: {filepath}",
            'errors': []
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'filepath': filepath,
            'message': f"Error reading file: {str(e)}",
            'errors': []
        }


def print_validation_report(result: Dict) -> None:
    """Pretty prints the validation results"""
    
    print("=" * 80)
    print("TRACKING ANNOTATION VALIDATION REPORT")
    print("=" * 80)
    
    print(f"\nFile: {result['filepath']}")
    print(f"Status: {result['status']}")
    
    if 'message' in result:
        print(f"Error: {result['message']}")
    else:
        print(f"Total lines processed: {result['total_lines']}")
        print(f"Valid lines: {result['valid_lines']}")
        
        if 'chimp_summary' in result:
            print(f"\nChimpanzee annotation counts:")
            for chimp, count in result['chimp_summary'].items():
                print(f"  - {chimp}: {count} entries")
        
        if result['errors']:
            print(f"\n⚠️  Found {len(result['errors'])} error(s):\n")
            
            for i, error in enumerate(result['errors'], start=1):
                print(f"Error #{i}:")
                print(f"  Line number: {error['line']}")
                
                if 'chimp' in error:
                    print(f"  Chimpanzee: {error['chimp']}")
                
                print(f"  Type: {error['type']}")
                print(f"  Message: {error['message']}")
                print(f"  Content: {error['content']}")
                
                if 'expected' in error:
                    print(f"  Expected format: {error['expected']}")
                
                print()
        else:
            print("\n✅ No errors found! All annotations are correctly formatted.")
    
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    # Use PATH_annotation if provided, otherwise accept command line argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Using file from command line argument: {filepath}\n")
    else:
        filepath = PATH_annotation
        print(f"Using file from PATH_annotation variable: {filepath}\n")
    
    # Check if file exists
    if not Path(filepath).exists():
        print(f"❌ Error: File not found at '{filepath}'")
        print(f"Please update PATH_annotation at the top of the script or provide the file path as an argument:")
        print(f"  python validate_tracking_annotations.py <filepath>")
        sys.exit(1)
    
    result = validate_tracking_file(filepath)
    print_validation_report(result)