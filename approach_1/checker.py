#!/usr/bin/env python3
"""
N3L Solution Validator - UPDATED
=================================
Handles both old and new solution file formats.
"""

import os
from pathlib import Path
from typing import List, Tuple
import sys
import re

def parse_solution_file(filepath: Path) -> Tuple[int, List[List[int]]]:
    """Parse solution file and extract n and grid (handles old and new formats)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract n - handle both formats
    n = None
    for line in lines:
        # New format: "# n=2"
        if line.startswith('# n='):
            n = int(line.split('=')[1].strip())
            break
        # Old format: "# N3L Solution: n=2, points=4"
        match = re.search(r'n=(\d+)', line)
        if match:
            n = int(match.group(1))
            break
    
    if n is None:
        raise ValueError(f"Could not find n in {filepath}")
    
    # Extract grid - handle both formats
    grid = []
    in_grid = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # New format: explicit "# Grid (0/1):" marker
        if line == '# Grid (0/1):':
            in_grid = True
            continue
        
        # Old format: grid starts after header, before coordinates
        if not in_grid and line and not line.startswith('#'):
            in_grid = True
        
        if in_grid:
            # Stop at coordinates or another header
            if line.startswith('#'):
                break
            if line == '':
                continue
                
            # Try to parse as grid row
            try:
                row = [int(x) for x in line.split()]
                if len(row) == n:
                    grid.append(row)
                elif len(row) > 0:
                    # Invalid row length
                    break
            except ValueError:
                # Not a number line, stop
                break
    
    if len(grid) != n:
        raise ValueError(f"Grid size mismatch in {filepath}: expected {n}x{n}, got {len(grid)}x{len(grid[0]) if grid else 0}")
    
    return n, grid

def is_collinear(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> bool:
    """Check if three points are collinear using triangle area formula."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Area = 0 means collinear
    area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    return area == 0

def get_points(grid: List[List[int]]) -> List[Tuple[int, int]]:
    """Extract all points (1s) from grid."""
    n = len(grid)
    points = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                # Store as 1-indexed to match coordinate output
                points.append((i+1, j+1))
    return points

def count_violations(grid: List[List[int]]) -> Tuple[int, List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]]:
    """Count collinear violations in grid."""
    points = get_points(grid)
    violations = []
    
    # Check all triplets
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    violations.append((points[i], points[j], points[k]))
    
    return len(violations), violations

def validate_solution(filepath: Path, verbose: bool = False) -> bool:
    """Validate a single solution file."""
    try:
        n, grid = parse_solution_file(filepath)
        points = get_points(grid)
        target = 2 * n
        
        # Check point count
        if len(points) != target:
            print(f"❌ {filepath.name}: Wrong point count (expected {target}, got {len(points)})")
            return False
        
        # Check violations
        num_violations, violations = count_violations(grid)
        
        if num_violations == 0:
            if verbose:
                print(f"✅ {filepath.name}: n={n}, points={len(points)}, violations=0")
            return True
        else:
            print(f"❌ {filepath.name}: n={n}, points={len(points)}, violations={num_violations}")
            if verbose and violations:
                print(f"   First violation: {violations[0]}")
            return False
            
    except Exception as e:
        print(f"⚠️  {filepath.name}: Error parsing - {e}")
        return False

def main():
    solutions_dir = Path("solutions")
    
    if not solutions_dir.exists():
        print(f"Error: {solutions_dir} directory not found")
        sys.exit(1)
    
    # Find all .txt files
    solution_files = list(solutions_dir.rglob("*.txt"))
    
    if not solution_files:
        print(f"No solution files found in {solutions_dir}")
        sys.exit(1)
    
    print(f"Found {len(solution_files)} solution files")
    print("="*60)
    
    # Validate each file
    results = {}
    valid_count = 0
    invalid_count = 0
    
    for filepath in sorted(solution_files):
        # Get n from directory name
        n_dir = filepath.parent.name
        
        if n_dir not in results:
            results[n_dir] = {'valid': [], 'invalid': []}
        
        is_valid = validate_solution(filepath, verbose=True)
        
        if is_valid:
            results[n_dir]['valid'].append(filepath.name)
            valid_count += 1
        else:
            results[n_dir]['invalid'].append(filepath.name)
            invalid_count += 1
    
    # Summary
    print("="*60)
    print("SUMMARY BY GRID SIZE:")
    print("="*60)
    
    for n_dir in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        valid = len(results[n_dir]['valid'])
        invalid = len(results[n_dir]['invalid'])
        total = valid + invalid
        
        status = "✅" if invalid == 0 else "⚠️"
        print(f"{status} n={n_dir}: {valid}/{total} valid")
        
        if invalid > 0:
            print(f"   Invalid files: {results[n_dir]['invalid']}")
    
    print("="*60)
    print(f"TOTAL: {valid_count} valid, {invalid_count} invalid")
    
    if invalid_count > 0:
        print("\n⚠️  WARNING: Found invalid solutions!")
        sys.exit(1)
    else:
        print("\n✅ All solutions validated successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()