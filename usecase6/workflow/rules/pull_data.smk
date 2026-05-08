rule pull_data:
    input:
        token = "pull_data/.token.txt",
    output:
        directory(f"{WORK}/pulled_data/")
    conda: "../../pull_data/environment.yml"
    shell:
        """
        python3 - <<'PYEOF'
import coscine, os
with open("{input.token}") as f:
    token = f.read().strip()
client  = coscine.ApiClient(token)
res     = client.project("showmehow_usecase5").resource("Field Data")
os.makedirs("{output}", exist_ok=True)
for file in res.files():
    file.download(path="{output}/")
PYEOF
        """
