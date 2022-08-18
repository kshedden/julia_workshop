using Downloads

rm("data", recursive = true, force = true)
mkdir("data")

base = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018"

files = ["DEMO_J.XPT", "BPX_J.XPT", "BMX_J.XPT", "ALQ_J.XPT"]

for f in files
    print("Downloading $f...")
    Downloads.download("$(base)/$(f)", "data/$(f)")
    println("done")
end
