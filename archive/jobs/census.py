import sys,torch,numpy as np
c=torch.load(sys.argv[1],map_location="cpu",weights_only=False);sp=c["splats"]
S=torch.exp(sp["scales"]);o=torch.sigmoid(sp["opacities"]);S=S[o>0.05].double().numpy()
Ss=np.sort(S,axis=1)[:,::-1]
print(f"   CENSUS N={len(S):,}  s2/s1 {np.median(Ss[:,1]/Ss[:,0]):.4f}  "
      f"s2/s3 {np.median(Ss[:,1]/np.maximum(Ss[:,2],1e-30)):.1f}  "
      f"s1/s3 {np.median(Ss[:,0]/np.maximum(Ss[:,2],1e-30)):.1f}",flush=True)
