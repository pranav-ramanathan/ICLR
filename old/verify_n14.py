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
print(f"Total points: {len(points)} (expected 28)")

def get_lines(n):
    lines = []
    for r in range(n):
        lines.append([(r, c) for c in range(n)])
    for c in range(n):
        lines.append([(r, c) for r in range(n)])
    from math import gcd
    for dr in range(-n+1, n):
        for dc in range(-n+1, n):
            if dr == 0 and dc == 0:
                continue
            if gcd(abs(dr), abs(dc)) != 1:
                continue
            for sr in range(n):
                for sc in range(n):
                    line = []
                    r, c = sr, sc
                    while 0 <= r < n and 0 <= c < n:
                        line.append((r, c))
                        r += dr
                        c += dc
                    if len(line) >= 3 and line not in lines:
                        lines.append(line)
    return lines

lines = get_lines(n)
print(f"Total lines: {len(lines)}")

violations = 0
for line in lines:
    count = sum(1 for (r, c) in line if board[r][c] == 1)
    if count >= 3:
        print(f"VIOLATION: {count} points on line starting {line[0]}")
        violations += 1

print(f"\n{'='*50}")
if violations == 0:
    print("VALID SOLUTION - No violations found!")
else:
    print(f"INVALID - {violations} violations")
print(f"{'='*50}")
