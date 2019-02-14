import os

def run(cmd):
    print(cmd)
    os.system(cmd)

print("--- Clearing buffers ---")
# Delete buffers if exist
run('dada_db -k dada -d')
run('dada_db -k eaea -d')

for sid in ('dbdisk', 'requant', 'udp_capture'):
    run('tmux kill-session -t {sid}'.format(sid=sid))

# Create buffers
print("\n--- Creating buffers ---")
buf_sz = 268435456
run('dada_db -k dada -n 64 -b {bs}'.format(bs=buf_sz))
run('dada_db -k eaea -n 4 -b {bs}'.format(bs=buf_sz/2))

#Build pipeline
print("\n--- Starting pipeline ---")
run("tmux new -d -s dbdisk 'dada_dbdisk -b 4 -k eaea -D /datax/PKSUWL'")
run("tmux new -d -s requant './bl_requant_16b_to_8b -b 2 dada eaea'")
run("tmux new -d -s udp_capture 'uwb_udpdb -v -k dada -b 1 -f dualvdif /home/obs/pks_seti/uwl_control/dada_headers/blc00_mcast_startup.cfg; sleep 10'")
