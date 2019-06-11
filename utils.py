
def read_data(PATH):
    #format: name,gender,_____
    file = open(PATH)
    data=[]
    for line in file:
        line = line.split(',')
        name = line[0].lower()
        gend = 1 if line[1] == 'M' else 0
        data.append([name,gend])
    return data

def get_chars():
    chars = {'<PAD>':0}
    for i in range(0,26):
        char = chr( ord('a') + i )
        chars[char] = i + 1
    return chars

def conv2vec(name,no_chars):
    # starting symbol + chars + padding + end 
    chars = get_chars()
    vec = []
    for char in name:
        index = chars[char]
        vec.append(index)
    if len(vec) < no_chars:
        diff = no_chars - len(vec) 
        vec+= [0]*(diff)
    return vec
    





