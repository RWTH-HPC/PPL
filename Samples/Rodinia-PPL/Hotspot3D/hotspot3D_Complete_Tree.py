from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('bc9n6f4c4AXeeU6iBz2g', 'Main')
dot.node('nA6trfiJKSXYSRyK5ZHF', 'Expression')
dot.node('orr1a3n4ixDYUmG9KWSa', 'Call: init_List', style="filled", fillcolor="green")
dot.node('nsCFX3puieKa3iEeZ7tr', 'Expression')
dot.node('QNT2yZzGQbxPFbREVhgX', 'Call: init_List', style="filled", fillcolor="green")
dot.node('QSrlQkLFp0aRj52du1RA', 'Expression')
dot.node('iundsAFdnGUhufr50A9Y', 'Call: init_List', style="filled", fillcolor="green")
dot.node('PeQJOYGN22RL4hIxSasu', 'Simple Expression')
dot.node('Xp1Zxnpa1Dbu8S3QTjdp', 'Expression')
dot.node('eTTVdRtaQrPblvvUeiOr', 'Call: compute_tran_temp', style="filled", fillcolor="green")
dot.node('pZF29T8bGRZH2vLy9ldm', 'Expression')
dot.node('ZiOlQTOg4rcbMFGeupME', 'Call: init_List', style="filled", fillcolor="green")
dot.node('sfAnIeCJUVet5J3omRLN', 'Expression')
dot.node('WucXIPuiQx4WZF8gksUC', 'Call: init_List', style="filled", fillcolor="green")
dot.node('XGxlYyqIUSWKIKSgHoSZ', 'Stencil: add_padding_1', style="filled", fillcolor="red")
dot.node('IhQrEXnI0JZ1R2X5YdBJ', 'Expression')
dot.node('dPC5OSvnul0nqMnxmfL9', 'Simple Expression')
dot.node('MkapAuMZchyJp8CkTiUv', 'Simple Expression')
dot.node('vg3V518UWLJCU4dxq8WE', 'For-Loop')
dot.node('s2Q9ypiOIyAZ29VDkifM', 'Expression')
dot.node('2igcyPip5jmfpWe3llWv', 'Expression')
dot.node('9LLmzKCGWQ3GUD3jcWZF', 'Expression')
dot.node('povIOv0Rdg3GO7XYxcq0', 'Stencil: side_north', style="filled", fillcolor="red")
dot.node('7LvS4V5NEGhYPIaoBeIc', 'Expression')
dot.node('PSP0OmVpTdEGgVQDFGCY', 'Simple Expression')
dot.node('loeZ34DYcb2spxb6TThp', 'Stencil: side_south', style="filled", fillcolor="red")
dot.node('JeoX2e6qxabxt5162oEc', 'Expression')
dot.node('QqLdFJW7EkWR2v9ZJk0R', 'Simple Expression')
dot.node('G9cakkSV4Sh1Ez83yG3S', 'Stencil: side_east', style="filled", fillcolor="red")
dot.node('1HzZK09SEXZr5uhYTVGM', 'Expression')
dot.node('3HJdiCUqlJ9exotAlPOy', 'Simple Expression')
dot.node('9tGYD3XZn73fUHjWSULK', 'Stencil: side_west', style="filled", fillcolor="red")
dot.node('m9V0CIZrZsHpzelol6Y5', 'Expression')
dot.node('oFIlZXSrH14tGhXfYnhi', 'Simple Expression')
dot.node('1ygWvzOxR6vTP7CKyqh5', 'Stencil: side_top', style="filled", fillcolor="red")
dot.node('C0thCZt2j7RNDnfF6U61', 'Expression')
dot.node('7p8aZblKHlVYpGCnRyRh', 'Simple Expression')
dot.node('TaTvkC8OqQd2qn7cUvWG', 'Stencil: side_bottom', style="filled", fillcolor="red")
dot.node('5KsktJHXSxmvvhirizCg', 'Expression')
dot.node('RF1lDbi1Rvn1ZzMySZqh', 'Simple Expression')
dot.node('u2F7FuUDdi6ZAn8pdjII', 'Stencil: single_iteration', style="filled", fillcolor="red")
dot.node('g1iz3Q7X1Q73YGjR249F', 'Expression')
dot.node('VUmzJAEIb9E2nUXagXRR', 'Simple Expression')
dot.node('Fulli3ScIP1OL1q7LchJ', 'Stencil: copy', style="filled", fillcolor="red")
dot.node('Zadk2iBK3gaVkacqBkS9', 'Expression')
dot.node('ocXpyJVWi7aQOxi1XuuB', 'Simple Expression')
dot.node('y0sA8GgUhcgU8Mrnhcd4', 'Return')
dot.node('5V5XL6ZuXbB9HKg5UZFr', 'Expression')
dot.node('H1gGkeFLOVW8m5H44iNo', 'Simple Expression')
dot.node('iM0A7lP5UI4ZTFDi6D4D', 'Return')
dot.node('szXgsjkL35v7dpPYRiNi', 'Expression')
dot.edge('bc9n6f4c4AXeeU6iBz2g', 'nA6trfiJKSXYSRyK5ZHF')
dot.edge('bc9n6f4c4AXeeU6iBz2g', 'nsCFX3puieKa3iEeZ7tr')
dot.edge('bc9n6f4c4AXeeU6iBz2g', 'QSrlQkLFp0aRj52du1RA')
dot.edge('bc9n6f4c4AXeeU6iBz2g', 'PeQJOYGN22RL4hIxSasu')
dot.edge('bc9n6f4c4AXeeU6iBz2g', 'Xp1Zxnpa1Dbu8S3QTjdp')
dot.edge('bc9n6f4c4AXeeU6iBz2g', 'H1gGkeFLOVW8m5H44iNo')
dot.edge('bc9n6f4c4AXeeU6iBz2g', 'iM0A7lP5UI4ZTFDi6D4D')
dot.edge('nA6trfiJKSXYSRyK5ZHF', 'orr1a3n4ixDYUmG9KWSa')
dot.edge('nsCFX3puieKa3iEeZ7tr', 'QNT2yZzGQbxPFbREVhgX')
dot.edge('QSrlQkLFp0aRj52du1RA', 'iundsAFdnGUhufr50A9Y')
dot.edge('Xp1Zxnpa1Dbu8S3QTjdp', 'eTTVdRtaQrPblvvUeiOr')
dot.edge('eTTVdRtaQrPblvvUeiOr', 'pZF29T8bGRZH2vLy9ldm')
dot.edge('eTTVdRtaQrPblvvUeiOr', 'sfAnIeCJUVet5J3omRLN')
dot.edge('eTTVdRtaQrPblvvUeiOr', 'XGxlYyqIUSWKIKSgHoSZ')
dot.edge('eTTVdRtaQrPblvvUeiOr', 'MkapAuMZchyJp8CkTiUv')
dot.edge('eTTVdRtaQrPblvvUeiOr', 'vg3V518UWLJCU4dxq8WE')
dot.edge('eTTVdRtaQrPblvvUeiOr', 'y0sA8GgUhcgU8Mrnhcd4')
dot.edge('pZF29T8bGRZH2vLy9ldm', 'ZiOlQTOg4rcbMFGeupME')
dot.edge('sfAnIeCJUVet5J3omRLN', 'WucXIPuiQx4WZF8gksUC')
dot.edge('XGxlYyqIUSWKIKSgHoSZ', 'IhQrEXnI0JZ1R2X5YdBJ')
dot.edge('XGxlYyqIUSWKIKSgHoSZ', 'dPC5OSvnul0nqMnxmfL9')
dot.edge('vg3V518UWLJCU4dxq8WE', 's2Q9ypiOIyAZ29VDkifM')
dot.edge('vg3V518UWLJCU4dxq8WE', '2igcyPip5jmfpWe3llWv')
dot.edge('vg3V518UWLJCU4dxq8WE', '9LLmzKCGWQ3GUD3jcWZF')
dot.edge('vg3V518UWLJCU4dxq8WE', 'povIOv0Rdg3GO7XYxcq0')
dot.edge('vg3V518UWLJCU4dxq8WE', 'loeZ34DYcb2spxb6TThp')
dot.edge('vg3V518UWLJCU4dxq8WE', 'G9cakkSV4Sh1Ez83yG3S')
dot.edge('vg3V518UWLJCU4dxq8WE', '9tGYD3XZn73fUHjWSULK')
dot.edge('vg3V518UWLJCU4dxq8WE', '1ygWvzOxR6vTP7CKyqh5')
dot.edge('vg3V518UWLJCU4dxq8WE', 'TaTvkC8OqQd2qn7cUvWG')
dot.edge('vg3V518UWLJCU4dxq8WE', 'u2F7FuUDdi6ZAn8pdjII')
dot.edge('vg3V518UWLJCU4dxq8WE', 'Fulli3ScIP1OL1q7LchJ')
dot.edge('povIOv0Rdg3GO7XYxcq0', '7LvS4V5NEGhYPIaoBeIc')
dot.edge('povIOv0Rdg3GO7XYxcq0', 'PSP0OmVpTdEGgVQDFGCY')
dot.edge('loeZ34DYcb2spxb6TThp', 'JeoX2e6qxabxt5162oEc')
dot.edge('loeZ34DYcb2spxb6TThp', 'QqLdFJW7EkWR2v9ZJk0R')
dot.edge('G9cakkSV4Sh1Ez83yG3S', '1HzZK09SEXZr5uhYTVGM')
dot.edge('G9cakkSV4Sh1Ez83yG3S', '3HJdiCUqlJ9exotAlPOy')
dot.edge('9tGYD3XZn73fUHjWSULK', 'm9V0CIZrZsHpzelol6Y5')
dot.edge('9tGYD3XZn73fUHjWSULK', 'oFIlZXSrH14tGhXfYnhi')
dot.edge('1ygWvzOxR6vTP7CKyqh5', 'C0thCZt2j7RNDnfF6U61')
dot.edge('1ygWvzOxR6vTP7CKyqh5', '7p8aZblKHlVYpGCnRyRh')
dot.edge('TaTvkC8OqQd2qn7cUvWG', '5KsktJHXSxmvvhirizCg')
dot.edge('TaTvkC8OqQd2qn7cUvWG', 'RF1lDbi1Rvn1ZzMySZqh')
dot.edge('u2F7FuUDdi6ZAn8pdjII', 'g1iz3Q7X1Q73YGjR249F')
dot.edge('u2F7FuUDdi6ZAn8pdjII', 'VUmzJAEIb9E2nUXagXRR')
dot.edge('Fulli3ScIP1OL1q7LchJ', 'Zadk2iBK3gaVkacqBkS9')
dot.edge('Fulli3ScIP1OL1q7LchJ', 'ocXpyJVWi7aQOxi1XuuB')
dot.edge('y0sA8GgUhcgU8Mrnhcd4', '5V5XL6ZuXbB9HKg5UZFr')
dot.edge('iM0A7lP5UI4ZTFDi6D4D', 'szXgsjkL35v7dpPYRiNi')



print(dot.source)
dot.render('hotspot3D_Complete_Tree.gv', view=True)