# we will implement the hungarian (the base algorithm) assignment algorithm to match students with the presentation subject two students per subject
import ranking
import comb
num_pt = 5
options = {
    1: "intro",
    2: "partie 1",
    3: "partie 2",
    4: "partie 3",
    5: "partie 4",
}
num_students = 10

students = ["haitham", "wissal", "amine", "hamza", "annas", "sara", "salssabil", "bilal", "wijdane", "waly"]

# number of students to be assigned to each subject
num_students_per_subject = 2

prefrences = []

def get_key_by_value(my_dict, value_to_find):
    # Iterates through the dictionary and returns the key that matches the value
    for key, value in my_dict.items():
        if value == value_to_find:
            return key
    return None  # Return None if the value is not found

for student in range(num_students):
    pr = [0] * (num_pt )
    r = ranking.interactive_ranking(list(options.values()))
    for i, choice in enumerate(r):
        pr[get_key_by_value(options, choice)-1] = 5 - i
    prefrences.append(pr)

combinations = list(comb.generate_combinations(range(num_students), num_pt))

best_comb = 0
best_satisfaction = 0
for comb_i, i in enumerate(combinations):
    satisfaction = 0
    for j in range(len(i)):
        satisfaction += prefrences[i[j][0]][j] + prefrences[i[j][1]][j]
    if satisfaction > best_satisfaction:
        best_satisfaction = satisfaction
        best_comb = comb_i

def print_comb(comb_idx):
    combination = combinations[comb_idx]
    for i in range(len(combination)):
        print(f"Student {students[combination[i][0]]} and Student {students[combination[i][1]]} are assigned to {options[i+1]}")

print('the combinations score is ', best_satisfaction)
print('best combinations is:')
print_comb(best_comb)
