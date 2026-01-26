# orientation.py

import numpy as np

def orient_v_structures(G, sepset):
    p = G.shape[0]
    for x in range(p):
        for y in range(p):
            if x == y or not G[x, y]:
                continue
            for z in range(p):
                if z in (x, y):
                    continue
                if G[x, z] and G[y, z] and not G[x, y]:
                    # unshielded triple x - z - y
                    if sepset[x][y] is None or z not in sepset[x][y]:
                        # orient x -> z <- y
                        G[x, z] = 1
                        G[z, x] = 0
                        G[y, z] = 1
                        G[z, y] = 0
    return G


def apply_orientation_rules(G):
    changed = True
    p = G.shape[0]

    while changed:
        changed = False

        # Rule 2: orient chains
        for x in range(p):
            for y in range(p):
                if G[x, y] == 1 and G[y, x] == 0:  # x -> y
                    for z in range(p):
                        if G[y, z] == 1 and G[z, y] == 1 and G[x, z] == 0 and G[z, x] == 0:
                            # y - z and x -> y and x not adjacent to z
                            G[y, z] = 1
                            G[z, y] = 0
                            changed = True

        # Rule 3: avoid new colliders
        for x in range(p):
            for y in range(p):
                if G[x, y] == 1 and G[y, x] == 1:  # x - y
                    for z in range(p):
                        if G[x, z] == 1 and G[z, x] == 0 and G[y, z] == 1 and G[z, y] == 1:
                            # x -> z and y - z
                            G[y, z] = 1
                            G[z, y] = 0
                            changed = True

        # Rule 4: avoid cycles
        for x in range(p):
            for y in range(p):
                if G[x, y] == 1 and G[y, x] == 0:  # x -> y
                    for z in range(p):
                        if G[y, z] == 1 and G[z, y] == 1 and G[x, z] == 0 and G[z, x] == 0:
                            G[y, z] = 1
                            G[z, y] = 0
                            changed = True

    return G


def orient_edges(G, sepset):
    G = orient_v_structures(G, sepset)
    G = apply_orientation_rules(G)
    return G
