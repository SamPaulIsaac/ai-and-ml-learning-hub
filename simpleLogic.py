# Simple AI: Guess if a person is an adult based on age

def classify_age(age):
    if age < 18:
        return "Child"
    else:
        return "Adult"

# Example
ages = [5, 17, 18, 25, 30]
for age in ages:
    print(f"Age {age}: {classify_age(age)}")
