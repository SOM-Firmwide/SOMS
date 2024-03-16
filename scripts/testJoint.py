from soms.datastructures import Joint

a = Joint(1,2,3)

a.restraint = [[True, True, True]]

print(a.restraint)