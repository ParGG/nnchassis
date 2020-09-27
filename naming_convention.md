# Modules
Modules should have short, all-lowercase names. Underscores can be used in the module name if it improves readability. Python packages should also have short, all-lowercase names, although the use of underscores is discouraged.

# Classes
Class names should normally use the CapWords convention.

# Functions and local variable names
lowercase, with words separated by underscores as necessary to improve readability

# How to come up with names
- Variable names are not full descriptors
First, do not think of variable names as full descriptors of their content. Names should be clear mainly to allow to keep track of where they come from and only then can they give a bit about the content.

    # This is very descriptive, but we can infer that by looking at the expression
    students_with_grades_above_90 = filter(lambda s: s.grade > 90, students)

    # This is short and still allows the reader to infer what the value was
    good_students = filter(lambda s: s.grade > 90, students)


- Put details in comments and docstrings
    # We feel we need a long variable description because the function is not documented
    def get_good_students(students):
        return filter(lambda s: s.grade > 90, students)

    students_with_grades_above_90 = get_good_students(students)


    # If the reader is not sure about what get_good_students returns,
    # their reflex should be to look at the function docstring
    def get_good_students(students):
        """Return students with grade above 90"""
        return filter(lambda s: s.grade > 90, students)

    # You can optionally comment here that good_students are the ones with grade > 90
    good_students = get_good_students(students)

- Too specifc name might mean too specific code
If you feel you need a very specific name for a function, it might be that the function is too specific itself.

    # Very long name because very specific behaviour
    def get_students_with_grade_above_90(students):
        return filter(lambda s: s.grade > 90, students)

    # Adding a level of abstraction shortens our method name here
    def get_students_above(grade, students):
        return filter(lambda s: s.grade > grade, students)

    # What we mean here is very clear and the code is reusable
    good_students = get_students_above(90, students)

- Keep short scopes for quick lookup
Encapsulate logic in shorter scopes. This way, you don't need to give that much detail in the variable name, as it can be looked up quickly a few lines above. A rule of thumb is to make your functions fit in your IDE without scrolling and encapsulate some logic in a new function if you go beyond that.

    # line 6
    names = ['John', 'Jane']

    ... # Hundreds of lines of code

    # line 371
    # Wait what was names again?
    if 'Kath' in names:
        ...

Here I lost track of what names was and scrolling up will make me lose track of even more. This is better:
    # line 6
    names = ['John', 'Jane']

    # I encapsulate part of the logic in another function to keep scope short
    x = some_function()
    y = maybe_a_second_function(x)

    # line 13
    # Sure I can lookup names, it is right above
    if 'Kath' in names:
        ...

- Spend time thinking about readability
The last but not least is to devote time thinking about how you can make your code more readable. This is as important as the time you spend thinking about the logic and code optimization. The best code is the code that everyone can read, and thus everyone can improve.