import coscine

with open("token.txt", "rt") as fp:
	token = fp.read()
# You can now use token to intialize the coscine ApiClient!

print(token)
client = coscine.ApiClient(token)