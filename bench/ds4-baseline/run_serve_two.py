import os, subprocess, sys, threading
root='/home/likan/strata-latest'
ids13=open('/tmp/strata-5137/ds4_prompt13k_ids.txt').read().strip()
ids3=open('/tmp/strata-5137/ds4_prompt_ids.txt').read().strip()
env=dict(os.environ)
env.update({
 'DS4_ENGRAM_CACHE_MB':'2048','DS4_SETTLE_MS':'2000','DS4_IQ1S_ROWS':'1','DS4_IQ_OWN':'1',
 'DS4_HIP_PEER_AR':'1','DS4_STORE_GB':'0','DS4_COPY_HUGETLB':'1','DS4_COPY_PIN':'1','DS4_ATTN_V7':'1'})
cmd=[root+'/build-dsv41/strata-dsv41','--pack','/mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-XS-v2/maya-ds4-v2',
 '--gpus','0,1','--tp','2','--max-context','32768','--reserve-mib','768','--cpu-threads','20','--mmap-experts','off',
 '--experts','4','--prefill-chunk','8192','--dspark','on','--serve']
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=env,cwd=root)
lines=[]
def err():
    for ln in p.stderr:
        lines.append(ln.rstrip())
        if 'prompt:' in ln: print('ERR',ln.strip(),flush=True)
threading.Thread(target=err,daemon=True).start()
def read_until_done(tag):
    while True:
        ln=p.stdout.readline()
        if not ln: print('EOF',tag); return
        if ln.startswith('READY') or ln.startswith('DONE') or ln.startswith('ERR'):
            print(tag,ln.strip(),flush=True)
        if ln.startswith('DONE') or ln.startswith('ERR'):
            return ln
# wait READY
while True:
    ln=p.stdout.readline()
    if not ln: break
    if ln.startswith('READY'): print('READY',ln.strip()); break
p.stdin.write('GEN 1 '+ids3+'\n'); p.stdin.flush()
read_until_done('first3k')
p.stdin.write('GEN 1 '+ids13+'\n'); p.stdin.flush()
read_until_done('second13k')
p.stdin.write('QUIT\n'); p.stdin.flush()
p.wait(timeout=30)
print('rc',p.returncode)
