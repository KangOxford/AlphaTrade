# input = "/Users/kang/AlphaTrade/docs/source/papers/s5.rst"
# input = "/Users/kang/dotfiles/docs/source/_learn.rst"
input = "/Users/kang/ChatPaper/export/2023-04-01-17-Towards Realistic Market Simulations_ a Generative Adversarial.md"
output = "/Users/kang/AlphaTrade/docs/source/_towards.rst"

# Open the file and read the contents
with open(input, 'r') as file:
    text = file.read()[25:]
lines = text.splitlines()
lines[0] = lines[0]+"\n"+"="*20
text = "\n".join(lines)
print(text)

text = text.replace("\n","\n\n")
# Replace the existing formatting with the desired formatting, maintaining the indentation for list items
text = text.replace("6. Summary:\n", "\nSummary:\n-----\n")
text = text.replace("7. Background:\n", "\nBackground:\n-----\n")
text = text.replace("8. Methods:\n", "\nMethods:\n-----\n")
text = text.replace("9. Conclusion:\n", "\nConclusion:\n-----\n")

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
start_lines = ["   ", "Background:", "Summary:", "Methods:", "Conclusion:", "a.", "b.", "c.", "d.", "e.", "f.", "g.", "---", "===", ".. ", "  ","1.","2.","3.","4.","5."]
for line in lines:
    if line == "" or line.startswith(tuple(start_lines)):
        new_lines.append(line)
    else:
        new_lines.append("   * " + line)

# Join the new lines back into a single string
new_text = "\n".join(new_lines)

title = "Towards Realistic Market Simulations: a Generative Adversarial Networks Approach"
new_text = ".. "+input.split("/")[-1][:-4]+":\n\n"+title+"==========================\n\n" + new_text
print(new_text)

# Write the updated text back to the file
with open(output, 'w') as file:
    file.write(new_text)
