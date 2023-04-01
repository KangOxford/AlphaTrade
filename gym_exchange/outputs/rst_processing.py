input = "/Users/kang/AlphaTrade/docs/source/papers/s5.rst"

# read input file
with open("/Users/kang/AlphaTrade/docs/source/papers/s5.rst", "r") as file:
    data = file.readlines()

# initialize variables to keep track of section headers
summary = False
background = False
methods = False
conclusion = False

# loop through each line in the file
for i in range(len(data)):
    # check if current line is a section header
    if data[i].startswith("Summary:"):
        summary = True
    elif data[i].startswith("Background:"):
        background = True
    elif data[i].startswith("Methods:"):
        methods = True
    elif data[i].startswith("Conclusion:"):
        conclusion = True

    # add separator line and empty line after section header if found
    if summary and not data[i + 1].startswith("-"):
        data.insert(i + 1, "\n----------\n")
        summary = False
    elif background and not data[i + 1].startswith("-"):
        data.insert(i + 1, "\n----------\n")
        background = False
    elif methods and not data[i + 1].startswith("-"):
        data.insert(i + 1, "\n----------\n")
        methods = False
    elif conclusion and not data[i + 1].startswith("-"):
        data.insert(i + 1, "\n----------\n")
        conclusion = False

# write modified data back to file
with open("/Users/kang/AlphaTrade/docs/source/papers/s5.rst", "w") as file:
    file.writelines(data)
