# input = "/Users/kang/AlphaTrade/docs/source/papers/s5.rst"
# input = "/Users/kang/dotfiles/docs/source/_learn.rst"
input = "/Users/kang/AlphaTrade/docs/source/_learn.rst"

# Open the file and read the contents
with open(input, 'r') as file:
    text = file.read()

text = text.replace("\n","\n\n")
# Replace the existing formatting with the desired formatting, maintaining the indentation for list items
text = text.replace("Summary:\n", "\nSummary:\n-----\n")
text = text.replace("Background:\n", "\nBackground:\n-----\n")
text = text.replace("Methods:\n", "\nMethods:\n-----\n")
text = text.replace("Conclusion:\n", "\nConclusion:\n-----\n")

text = text.replace("(1):", "\n1.")
text = text.replace("(2):", "\n2.")
text = text.replace("(3):", "\n3.")
text = text.replace("(4):", "\n4.")

text = text.replace("a.", "\na.")
text = text.replace("b.", "\nb.")
text = text.replace("c.", "\nc.")
text = text.replace("d.", "\nd.")
text = text.replace("e.", "\ne.")

lines = text.split("\n")
new_lines = []
for line in lines:
    if line == "" or line.startswith("   ") or line.startswith("Background:") or line.startswith("Summary:") or line.startswith("Methods:") or line.startswith("Conclusion:") or line.startswith("a.") or line.startswith("b.") or line.startswith("c.") or line.startswith("d.") or line.startswith("e.") or line.startswith("f.") or line.startswith("g.") or line.startswith("---") or line.startswith("===") or line.startswith(".. ") or line.startswith("  "):
        new_lines.append(line)
    else:
        new_lines.append("   * " + line)

# Join the new lines back into a single string
new_text = "\n".join(new_lines)

print(new_text)

# Write the updated text back to the file
with open(input, 'w') as file:
    file.write(new_text)
