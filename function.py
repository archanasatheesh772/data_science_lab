def display(m1, m2, m3):
    total = m1 + m2 + m3
    percentage = total / 3

    if percentage >= 90:
        grade = "A"
    elif percentage >= 80:
        grade = "B"
    elif percentage >= 70:
        grade = "C"
    else:
        grade = "D"

    print("Total =", total)
    print("Percentage =", percentage)
    print("Grade =", grade)


m1 = int(input("Enter mark1: "))
m2 = int(input("Enter mark2: "))
m3 = int(input("Enter mark3: "))

display(m1, m2, m3)