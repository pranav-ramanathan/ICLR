#!/usr/bin/env python3

board = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
]

n = 14

points = [(r, c) for r in range(n) for c in range(n) if board[r][c] == 1]
print(f"Total points: {len(points)}")

from math import gcd

def get_lines(n):
    lines = []
    dirs = set()
    
    for dx in range(n):
        for dy in range(n):
            if dx == 0 and dy == 0:
                continue
            g = gcd(dx, dy)
            dxp, dyp = dx // g, dy // g
            if dxp > 0 or (dxp == 0 and dyp > 0):
                dirs.add((dxp, dyp))
    
    seen = set()
    for dx, dy in dirs:
        for i in range(n):
            for j in range(n):
                if 0 <= i - dx < n and 0 <= j - dy < n:
                    continue
                pts = []
                ii, jj = i, j
                while 0 <= ii < n and 0 <= jj < n:
                    pts.append((ii, jj))
                    ii += dx
                    jj += dy
                if len(pts) >= 3:
                    key = tuple(sorted(pts))
                    if key not in seen:
                        seen.add(key)
                        lines.append(pts)
    return lines

lines = get_lines(n)
print(f"Total lines: {len(lines)}")

violations = 0
for line in lines:
    count = sum(1 for (r, c) in line if board[r][c] == 1)
    if count >= 3:
        print(f"VIOLATION: {count} points on line starting {line[0]} direction ({line[1][0]-line[0][0]}, {line[1][1]-line[0][1]})")
        violations += 1

print(f"\nTotal violations: {violations}")
