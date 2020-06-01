'''import re

input = re.split("[.?]+", "what are you doing? cc. cl")

print(input)
'''
input = 'what are you doing? how old are you?. .lie'

inputArray = []

index = 0
temp = 0
lenInput = len(input)

print(lenInput)


while index < lenInput:
    if(input[index] == '.' or input[index] == '?'):
        print(temp, index)
        inputArray.append(input[temp: index + 1])
        temp = index + 1
        index += 1
        continue
    if(index == lenInput-1):
        inputArray.append(input[temp: index + 1])
    index += 1

'''for x in range(len(input) - 1):
    print(x)
    for y in range(x, len((input))):
        if(input[y] == '.' or input[y] == '?'):
            inputArray.append(input[x:y])
            inputArray.append(input[y])
            x = x + y
            break
'''

print(inputArray)
