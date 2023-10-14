def find_modified_max_argmax(L, f):
    l = [f(i) for i in L if type(i) == int]
    return tuple([max(l), l.index(max(l))]) if l else ()

def find_modified_max_argmax(L, f):
    M=()
    for x in L:
        if type(x) is int:
            M=max(M, (f(x), x))
    return M