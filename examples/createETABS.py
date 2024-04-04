from soms.connectors import ETABS

session = ETABS.from_instance()
print(session.client)
print(session.model)

frames = session.GetFrames()
joints = session.GetJoints()
areas = session.GetAreas()

for i in frames:
    print(i, frames[i])

for i in joints:
    print(i, joints[i])

for i in areas:
    print(i, areas[i], len(areas[i]))