#!/usr/bin/env python3
"""
Verify solution files for no-three-in-line problem.
Checks if claimed zero violations are accurate.
"""

import os
from pathlib import Path
from itertools import combinations
from collections import defaultdict

def read_solution_file(filepath):
    """Read a solution file and extract n and the board."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract n from filename (format: n_X_Y.txt)
    filename = os.path.basename(filepath)
    n = int(filename.split('_')[1])
    
    # Parse board - skip header lines, read until we hit the board
    board = []
    in_board = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Look for lines that contain only 0s and 1s (with spaces)
        if all(c in '01 ' for c in line) and ('0' in line or '1' in line):
            in_board = True
            row = [int(c) for c in line.split()]
            if row:  # Only add non-empty rows
                board.append(row)
    
    return n, board

def get_point_positions(board):
    """Get list of (row, col) positions where points exist."""
    points = []
    for i, row in enumerate(board):
        for j, val in enumerate(row):
            if val == 1:
                points.append((i, j))
    return points

def are_collinear(p1, p2, p3):
    """Check if three points are collinear using cross product."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # Cross product: (p2-p1) × (p3-p1) = 0 if collinear
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def count_violations(points):
    """Count number of collinear triples (3+ points on a line)."""
    if len(points) < 3:
        return 0
    
    # Group points by lines they form
    lines = defaultdict(set)
    
    # Check all triples
    for i, p1 in enumerate(points):
        for j in range(i + 1, len(points)):
            p2 = points[j]
            # Define line by first two points
            for k in range(j + 1, len(points)):
                p3 = points[k]
                if are_collinear(p1, p2, p3):
                    # Create canonical line representation
                    # Sort the three points to create a canonical form
                    sorted_points = tuple(sorted([p1, p2, p3]))
                    # But we need to track which line this is
                    # Use direction vector and a point
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Normalize direction
                    from math import gcd
                    g = gcd(abs(dx), abs(dy)) if dx or dy else 1
                    if g > 0:
                        dx, dy = dx // g, dy // g
                    # Make direction canonical (handle negative)
                    if dx < 0 or (dx == 0 and dy < 0):
                        dx, dy = -dx, -dy
                    
                    # Use point p1 and direction as line key
                    # But we need to account for different points on same line
                    # Calculate line equation: dy*x - dx*y = c
                    c = dy * p1[0] - dx * p1[1]
                    line_key = (dx, dy, c)
                    
                    lines[line_key].update([p1, p2, p3])
    
    # Count violations: for each line with k points, violations = C(k, 3)
    total_violations = 0
    for line_points in lines.values():
        k = len(line_points)
        if k >= 3:
            # Number of ways to choose 3 points from k points
            violations = k * (k - 1) * (k - 2) // 6
            total_violations += violations
    
    return total_violations

def verify_solution(filepath):
    """Verify a single solution file."""
    try:
        n, board = read_solution_file(filepath)
        
        if not board:
            return {
                'filename': os.path.basename(filepath),
                'n': n,
                'points': 0,
                'expected_points': 2 * n,
                'violations': 0,
                'valid': False,
                'error': 'Could not read board'
            }
        
        points = get_point_positions(board)
        num_points = len(points)
        expected_points = 2 * n
        violations = count_violations(points)
        
        # Valid if: correct number of points AND zero violations
        valid = (num_points == expected_points) and (violations == 0)
        
        return {
            'filename': os.path.basename(filepath),
            'n': n,
            'points': num_points,
            'expected_points': expected_points,
            'violations': violations,
            'valid': valid,
            'error': None
        }
    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'n': 0,
            'points': 0,
            'expected_points': 0,
            'violations': 0,
            'valid': False,
            'error': str(e)
        }

def main():
    # Files to check
    files_to_check = [
        'n_3_1036464.txt',
        'n_4_1036871.txt', 'n_4_1036136.txt', 'n_4_103599.txt', 
        'n_4_1033615.txt', 'n_4_103018.txt',
        'n_5_1013636.txt',
        'n_6_1014815.txt',
        'n_7_1015794.txt',
        'n_8_1015074.txt',
        'n_9_101636.txt',
        'n_10_103831.txt',
        'n_11_1041339.txt',
        'n_12_1045428.txt',
        'n_13_1646665.txt',
        'n_14_173806.txt',
    ]
    
    base_dir = Path('/Users/pranavr/Developer/Work/Research/ICLR')
    
    print("=" * 100)
    print("SOLUTION FILE VERIFICATION REPORT")
    print("=" * 100)
    print()
    
    results = []
    for filename in files_to_check:
        filepath = base_dir / filename
        if filepath.exists():
            result = verify_solution(filepath)
            results.append(result)
        else:
            results.append({
                'filename': filename,
                'n': 0,
                'points': 0,
                'expected_points': 0,
                'violations': 0,
                'valid': False,
                'error': 'File not found'
            })
    
    # Print detailed table
    print(f"{'Filename':<20} {'n':>3} {'Points':>7} {'Expected':>8} {'Violations':>11} {'Valid':>7} {'Error':<20}")
    print("-" * 100)
    
    valid_count = 0
    invalid_count = 0
    
    for result in results:
        valid_str = '✓ YES' if result['valid'] else '✗ NO'
        error_str = result['error'] if result['error'] else ''
        
        print(f"{result['filename']:<20} {result['n']:>3} {result['points']:>7} "
              f"{result['expected_points']:>8} {result['violations']:>11} "
              f"{valid_str:>7} {error_str:<20}")
        
        if result['valid']:
            valid_count += 1
        else:
            invalid_count += 1
    
    print("-" * 100)
    print()
    
    # Summary
    print("SUMMARY:")
    print(f"  Total files checked: {len(results)}")
    print(f"  Valid solutions (0 violations): {valid_count}")
    print(f"  Invalid solutions (has violations or errors): {invalid_count}")
    print()
    
    # List valid files
    if valid_count > 0:
        print("✓ VALID FILES (0 violations):")
        for result in results:
            if result['valid']:
                print(f"  - {result['filename']} (n={result['n']}, {result['points']} points)")
        print()
    
    # List invalid files
    if invalid_count > 0:
        print("✗ INVALID FILES:")
        for result in results:
            if not result['valid']:
                reason = result['error'] if result['error'] else f"{result['violations']} violations"
                point_issue = ""
                if result['points'] != result['expected_points'] and not result['error']:
                    point_issue = f" (points: {result['points']}/{result['expected_points']})"
                print(f"  - {result['filename']} (n={result['n']}): {reason}{point_issue}")
        print()
    
    print("=" * 100)

if __name__ == '__main__':
    main()
