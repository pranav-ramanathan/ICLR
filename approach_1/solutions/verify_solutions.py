#!/usr/bin/env python3
"""
Verification script for No-Three-In-Line problem solutions.

The No-Three-In-Line problem asks: what is the maximum number of points
that can be placed on an n×n grid such that no three points are collinear?

This script verifies that all solutions in the repository obey the rule:
- No three points lie on the same straight line (horizontal, vertical, or diagonal)
- Each solution has exactly 2n points on an n×n grid

References:
- https://en.wikipedia.org/wiki/No-three-in-line_problem
- Verified for n ≤ 52: N(n) = 2n (Achim Flammenkamp et al.)
"""

import os
import re
from pathlib import Path
from fractions import Fraction
from typing import List, Tuple, Set


def parse_solution(filepath: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Parse a solution file and extract n, target, and coordinates.
    Handles both old and new format files.
    
    Returns:
        (n, target, list of (row, col) coordinates)
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse n and target from header - try new format first
    n_match = re.search(r'# n=(\d+)', content)
    target_match = re.search(r'# target=(\d+)', content)
    
    if n_match:
        n = int(n_match.group(1))
        target = int(target_match.group(1)) if target_match else 2 * n
    else:
        # Try old format: "# N3L Solution: n=2, points=4"
        old_match = re.search(r'# N3L Solution: n=(\d+),\s*points=(\d+)', content)
        if old_match:
            n = int(old_match.group(1))
            target = int(old_match.group(2))
        else:
            raise ValueError(f"Could not find n in {filepath}")
    
    # Parse coordinates
    coords = []
    in_coords_section = False
    
    for line in content.split('\n'):
        if '# Coordinates' in line:
            in_coords_section = True
            continue
        
        if in_coords_section:
            # Match (row, col) format
            match = re.search(r'\((\d+),\s*(\d+)\)', line)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))
                coords.append((row, col))
    
    return n, target, coords


def check_three_collinear(points: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Check if any three points are collinear.
    
    Three points (x1,y1), (x2,y2), (x3,y3) are collinear if:
    (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1)
    
    Or equivalently, the area of the triangle they form is zero.
    
    Returns:
        List of triples of collinear points
    """
    collinear_triples = []
    n = len(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                
                # Check collinearity using cross product
                # Points are collinear if (p2-p1) × (p3-p1) = 0
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                
                # Cross product: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                
                if cross == 0:
                    collinear_triples.append((p1, p2, p3))
    
    return collinear_triples


def verify_solution(filepath: str) -> dict:
    """
    Verify a single solution file.
    
    Returns:
        Dictionary with verification results
    """
    result = {
        'filepath': filepath,
        'valid': True,
        'errors': [],
        'n': None,
        'num_points': None,
        'target': None,
        'collinear_triples': []
    }
    
    try:
        n, target, coords = parse_solution(filepath)
        result['n'] = n
        result['num_points'] = len(coords)
        result['target'] = target
        
        # Check number of points
        if len(coords) != target:
            result['errors'].append(
                f"Expected {target} points, found {len(coords)}"
            )
            result['valid'] = False
        
        # Check for duplicate points
        if len(coords) != len(set(coords)):
            duplicates = [p for p in set(coords) if coords.count(p) > 1]
            result['errors'].append(f"Duplicate points found: {duplicates}")
            result['valid'] = False
        
        # Check points are within grid bounds (1-indexed)
        for r, c in coords:
            if not (1 <= r <= n and 1 <= c <= n):
                result['errors'].append(
                    f"Point ({r}, {c}) is outside {n}x{n} grid"
                )
                result['valid'] = False
        
        # Check no three collinear
        collinear = check_three_collinear(coords)
        if collinear:
            result['collinear_triples'] = collinear
            result['errors'].append(
                f"Found {len(collinear)} collinear triples: {collinear[:3]}"
            )
            result['valid'] = False
        
    except Exception as e:
        result['errors'].append(f"Parse error: {str(e)}")
        result['valid'] = False
    
    return result


def main():
    """Main verification function."""
    solutions_dir = Path(__file__).parent
    
    # Find all solution files
    solution_files = []
    for subdir in solutions_dir.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            for sol_file in subdir.glob('sol_*.txt'):
                solution_files.append(sol_file)
    
    solution_files.sort()
    
    print("=" * 70)
    print("NO-THREE-IN-LINE SOLUTION VERIFICATION")
    print("=" * 70)
    print(f"Found {len(solution_files)} solution files to verify\n")
    
    # Verify all solutions
    results = []
    for sol_file in solution_files:
        result = verify_solution(str(sol_file))
        results.append(result)
    
    # Summary statistics
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total solutions: {len(results)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    
    # Group by n value
    by_n = {}
    for r in results:
        n = r['n']
        if n is None:
            continue  # Skip files that failed to parse
        if n not in by_n:
            by_n[n] = {'valid': 0, 'invalid': 0, 'errors': []}
        if r['valid']:
            by_n[n]['valid'] += 1
        else:
            by_n[n]['invalid'] += 1
            by_n[n]['errors'].extend(r['errors'])
    
    print(f"\n{'=' * 70}")
    print("RESULTS BY GRID SIZE (n)")
    print(f"{'=' * 70}")
    
    for n in sorted(by_n.keys()):
        stats = by_n[n]
        status = "✓ ALL VALID" if stats['invalid'] == 0 else f"✗ {stats['invalid']} INVALID"
        print(f"n={n:2d}: {stats['valid']:2d} valid, {stats['invalid']:2d} invalid - {status}")
    
    # Print details of invalid solutions
    if invalid_count > 0:
        print(f"\n{'=' * 70}")
        print("INVALID SOLUTION DETAILS")
        print(f"{'=' * 70}")
        
        for result in results:
            if not result['valid']:
                print(f"\n{result['filepath']}")
                print(f"  n={result['n']}, points={result['num_points']}, target={result['target']}")
                for error in result['errors']:
                    print(f"  ERROR: {error}")
    
    print(f"\n{'=' * 70}")
    print("VERIFICATION COMPLETE")
    print(f"{'=' * 70}")
    
    # Exit with error code if any invalid
    if invalid_count > 0:
        print(f"\n✗ {invalid_count} solution(s) failed verification!")
        exit(1)
    else:
        print("\n✓ All solutions passed verification!")
        exit(0)


if __name__ == '__main__':
    main()
