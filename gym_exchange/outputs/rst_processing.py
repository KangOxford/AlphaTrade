# input = "/Users/kang/AlphaTrade/docs/source/papers/s5.rst"
input = "/Users/kang/dotfiles/docs/source/_dyn.rst"

# Open the file and read the contents
with open(input, 'r') as file:
    text = file.read()

# Replace the existing formatting with the desired formatting, maintaining the indentation for list items
text = text.replace("Summary:\n(1):", "Summary:\n-----\n1.")
text = text.replace("(2):", "\n2.")
text = text.replace("(3):", "\n3.")
text = text.replace("(4):", "\n4.")

text = text.replace("Background:\na.", "Background:\n-----\na.")
text = text.replace("b.", "\nb.")
text = text.replace("c.", "\nc.")
text = text.replace("d.", "\nd.")
text = text.replace("e.", "\ne.")

text = text.replace("Methods:\na.", "Methods:\n-----\na.")
text = text.replace("b.", "\nb.")

text = text.replace("Conclusion:\na.", "Conclusion:\n-----\na.")
text = text.replace("b.", "\nb.")
text = text.replace("c.", "\nc.")
text = text.replace("   *", "      *")

# Write the updated text back to the file
with open(input, 'w') as file:
    file.write(text)
