# input = "/Users/kang/AlphaTrade/docs/source/papers/s5.rst"
# input = "/Users/kang/dotfiles/docs/source/_learn.rst"
# input = "/Users/kang/ChatPaper/export/2023-04-01-17-Towards Realistic Market Simulations_ a Generative Adversarial.md"
# output = "/Users/kang/AlphaTrade/docs/source/_towards.rst"

input = "/Users/kang/ChatPaper/export/2023-04-01-17-Learning to simulate realistic limit order book markets from.md"
output = "/Users/kang/AlphaTrade/docs/source/_learning.rst"

# Open the file and read the contents
with open(input, 'r') as file:
    text = file.read()[25:]
lines = text.splitlines()
lines[0] = lines[0]+"\n"+"="*20
text = "\n".join(lines)


#
# lines = text.splitlines()
# for line in lines:
#     line = line.lstrip()
# # Join the modified lines back into a single string
# text = ''.join(lines)
#
#
#


import re
# define the regular expression pattern
pattern = r'(\d+\.\s\w+:\s)'
replace = r'\1' + "\n" +'-'*20 + "\n\n"
text = re.sub(pattern, replace, text)

text = text.replace("7. Methods:","\n7. Methods:")
text = re.sub(r'- \((\d+)\):', r'\1.', text)
text = re.sub(r'\n', r'\n\n', text)
text = text.replace("\n===", "===")
text = text.replace("\n---", "---")
try:
    for i in [3,4,5]:
        tobe_replaced = '\n' * i
        while tobe_replaced in text:
            text = text.replace(tobe_replaced, "\n\n")
except:
    pass
text = ".. "+output.split("/")[-1][:-4]+":\n\n" + text
# Replace "digit. word:" with "digit-1. word:"
text = re.sub(r'(\d+)\. (\w+):', lambda match: f'{int(match.group(1))-1}. {match.group(2)}:', text)


print(text)

# Write the updated text back to the file
with open(output, 'w') as file:
    file.write(text)
