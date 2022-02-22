# %%
import csv
from collections import deque

#mydict = {}
dict_from_csv={}
with open('distances.csv', mode='r') as inp:
    reader = csv.reader(inp)
    for rows in reader:
        if rows[0] in dict_from_csv.keys():
            dict_from_csv[rows[0]]=(dict_from_csv[rows[0]],rows[0])
        else:
            break
print(dict_from_csv)

# %%
def list_to_adj_list(inp_list):
    adj_list={}
    while 1:
        adj_list.update({inp_list})
    

# %%

 
class Graph:
    def __init__(self, adjacency_lis):
        self.adjacency_lis = adjacency_lis
 
    def get_neighbors(self, v):
        return self.adjacency_lis[v]
 
    # Heuristic value is 0 for all nodes
    def h():
      return 0
 
    def a_star_algorithm(self, start, stop):
        open_lst = set([start])
        closed_lst = set([])

        poo = {}
        poo[start] = 0
 
        # par contains an adjac mapping of all nodes
        par = {}
        par[start] = start
 
        while len(open_lst) > 0:
            n = None

            for v in open_lst:
                if n == None or poo[v] + self.h(v) < poo[n] + self.h(n):
                    n = v;
 
            if n == None:
                print('Path does not exist!')
                return None

            if n == stop:
                reconst_path = []
 
                while par[n] != n:
                    reconst_path.append(n)
                    n = par[n]
 
                reconst_path.append(start)
 
                reconst_path.reverse()
 
                print('Bulunan yol : '.format(reconst_path))
                return reconst_path
            for (m, weight) in self.get_neighbors(n):

                if m not in open_lst and m not in closed_lst:
                    open_lst.add(m)
                    par[m] = n
                    poo[m] = poo[n] + weight
 

                else:
                    if poo[m] > poo[n] + weight:
                        poo[m] = poo[n] + weight
                        par[m] = n
 
                        if m in closed_lst:
                            closed_lst.remove(m)
                            open_lst.add(m)
 

            open_lst.remove(n)
            closed_lst.add(n)
 
        print('Yol yok')
        return None


# %%



