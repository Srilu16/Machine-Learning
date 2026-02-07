A = [[1, 2, 3],
     [2, 2, 1],
     [3, 3, 5]]
n=len(A)
m=int(input("Enter value of m : "))
result=[]
for i in range(n):
    row=[]
    for j in range(n):
        if i == j:
            row.append(1)
        else:
            row.append(0)
    result.append(row)
for k in range(m):
    temp = [[0]*n for i in range(n)]
    for i1 in range(n):
        for j1 in range(n):
            for k1 in range(n):
                temp[i1][j1]+=result[i1][k1]*A[k1][j1]
    result=temp

print("A ^ ",m,"-")
print(result)
        
