import pickle
import os

def load_pickle(path, class_name):
    with open(path + "\\" + class_name + '.pkl', 'rb') as f:
        A = pickle.load(f)
        f.close()
    return A

def save_class_as_pickle(path, Obj):

    if hasattr(Obj, "name"):
        with open(path + "\\" + Obj.name + '.pkl', 'wb') as f:
            pickle.dump(Obj, f)
            f.close()
    else:
        name_taken = True
        number = 0
        while name_taken:
            name = "unnamed{}.pkl".format(number)
            if name in os.listdir(path=path):
                continue
            else:
                with open(path + "\\" + name, 'wb') as f:
                    pickle.dump(Obj, f)
                    f.close()
                name_taken = False
