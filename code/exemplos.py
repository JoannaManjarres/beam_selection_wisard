
produto = ''
carrinho = []



lista = [1,2,3]

num1, num2, num3 = lista



for i, v in enumerate(lista):
    print(i, v)

for num in lista:
    print(num)

for i in range(len(lista)):
    print(lista[i])

while produto!='sair':
    print("adicione um produto ou digite sair")
    produto = input()
    if produto!= 'sair':
        carrinho.append(produto)


for produto in carrinho:
    print(produto)

