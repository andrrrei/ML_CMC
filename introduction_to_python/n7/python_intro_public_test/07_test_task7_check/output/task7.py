def find_modified_max_argmax(L, f):
    l = [f(i) for i in L if type(i) == int]
    m = max(l)
    return tuple([m, l.index(m)]) if l else ()