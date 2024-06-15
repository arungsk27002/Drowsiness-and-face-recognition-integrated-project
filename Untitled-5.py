def maximumelement(arr):
    index=0
    maximum=arr[0]
    while(index<len(arr)-1):
        if ((maximum<arr[index+1]) and (index<len(arr)-1)):
            
            maximum=arr[index+1]
        index+=1
    return maximum
array=[1,2,3]
print(maximumelement(array))