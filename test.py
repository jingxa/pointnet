import  numpy as np
#from utils import pc_util
a = [1,1,1,1,1]
b = [2,2,2,2,2]
c = [3,3,3,3,3]
d = [a,b,c]

#vol = pc_util.point_cloud_to_volume(d,3)

# print(vol)
# e = [(a[i],b[i],c[i]) for i in range(5)]
# print(e)
# print("---------------------")
# v = np.array(e,dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
# print(v)
# print(v.shape)
# print("-----------------------")
# pc = np.array([[x,y,z] for x,y,z in v])
# print(pc)
# print(pc.shape)

e= np.array([a,b])
print(e)
print(e.shape)
print("----------------------")
v = np.lib.pad(e,((2,1),(4,3)),'constant',constant_values=3)
print(v)
print(v.shape)