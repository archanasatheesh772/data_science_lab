import matplotlib.pyplot as plt


x = [1, 2, 3, 4]
y = [5, 6, 7, 8]


plt.plot(x, y)


plt.savefig('my_plot.png')

print("Success! Your plot has been saved as 'my_plot.png' in your project folder.")