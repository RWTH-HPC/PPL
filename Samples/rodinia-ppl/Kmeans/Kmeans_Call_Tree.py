from graphviz import Digraph


dot = Digraph(comment='Abstract Pattern Tree')

dot  #doctest: +ELLIPSIS


dot.node('V4nfJLr7Ku9x2DpuctcG', 'Main')
dot.node('tXAsKtgD25E8znyne6fB', 'Call: init_List', style="filled", fillcolor="green")
dot.node('2xbBci2KQVWp3fo1uhoi', 'Call: init_List', style="filled", fillcolor="green")
dot.node('KFmX49050VbLhYvxOKek', 'Call: init_List', style="filled", fillcolor="green")
dot.node('9XWHHFpvYq0oR6nhptCp', 'Call: init_List', style="filled", fillcolor="green")
dot.node('JiSmJlZ1peiyz1fczzfi', 'Call: k_means', style="filled", fillcolor="green")
dot.node('WJOxa2xWdqQKpI1a4svh', 'Call: init_List', style="filled", fillcolor="green")
dot.node('GtgH1T0mreRxVlbroOHU', 'Call: init_List', style="filled", fillcolor="green")
dot.node('njM95F1Y77eS8QzOG5CP', 'Call: init_List', style="filled", fillcolor="green")
dot.node('CBfj2Qm66Sx25lt1LxK6', 'Map: copy', style="filled", fillcolor="red")
dot.node('QoNW9JNCYGbbOdtg7oUW', 'Map: determine_cemtroids', style="filled", fillcolor="red")
dot.node('MjqaSpyUxFHVx16zck1F', 'Call: assign_centroid', style="filled", fillcolor="green")
dot.node('W9bWEBOh4orkfBXq78QA', 'Map: update_centroids', style="filled", fillcolor="red")
dot.node('aLiPkV5rnq7MHEpJbxbt', 'Call: assigned_sum', style="filled", fillcolor="green")
dot.node('p6JxmGcOFhCmGrseUfQP', 'Call: assigned_sum', style="filled", fillcolor="green")
dot.node('dJnPhoaKGY0ufMOxNgFZ', 'Call: assigned_count', style="filled", fillcolor="green")
dot.edge('V4nfJLr7Ku9x2DpuctcG', 'tXAsKtgD25E8znyne6fB')
dot.edge('V4nfJLr7Ku9x2DpuctcG', '2xbBci2KQVWp3fo1uhoi')
dot.edge('V4nfJLr7Ku9x2DpuctcG', 'KFmX49050VbLhYvxOKek')
dot.edge('V4nfJLr7Ku9x2DpuctcG', '9XWHHFpvYq0oR6nhptCp')
dot.edge('V4nfJLr7Ku9x2DpuctcG', 'JiSmJlZ1peiyz1fczzfi')
dot.edge('JiSmJlZ1peiyz1fczzfi', 'WJOxa2xWdqQKpI1a4svh')
dot.edge('JiSmJlZ1peiyz1fczzfi', 'GtgH1T0mreRxVlbroOHU')
dot.edge('JiSmJlZ1peiyz1fczzfi', 'njM95F1Y77eS8QzOG5CP')
dot.edge('JiSmJlZ1peiyz1fczzfi', 'CBfj2Qm66Sx25lt1LxK6')
dot.edge('JiSmJlZ1peiyz1fczzfi', 'QoNW9JNCYGbbOdtg7oUW')
dot.edge('QoNW9JNCYGbbOdtg7oUW', 'MjqaSpyUxFHVx16zck1F')
dot.edge('JiSmJlZ1peiyz1fczzfi', 'W9bWEBOh4orkfBXq78QA')
dot.edge('W9bWEBOh4orkfBXq78QA', 'aLiPkV5rnq7MHEpJbxbt')
dot.edge('W9bWEBOh4orkfBXq78QA', 'p6JxmGcOFhCmGrseUfQP')
dot.edge('W9bWEBOh4orkfBXq78QA', 'dJnPhoaKGY0ufMOxNgFZ')



print(dot.source)
dot.render('Kmeans_Call_Tree.gv', view=True)