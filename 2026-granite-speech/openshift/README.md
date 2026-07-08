# OpenShift deployment (granite-speech-demo)

Deploys the frontend (Next.js) and backend (Pipecat/FastAPI) to the CPU cluster
`c111-e.us-east.containers.cloud.ibm.com`, project **`granite-speech-demo`**.
The Granite audio LLM stays on the GPU cluster; the backend calls it over HTTPS.

## Deploy

```bash
export HF_TOKEN=hf_...          # or put it (uncommented) in ../.env
./openshift/deploy.sh
```

`deploy.sh` is idempotent: creates the Secret/ConfigMaps, applies the manifests,
runs on-cluster binary builds from a minimal staged context (not the whole repo —
the old `oc` client ignores `.dockerignore` and would upload `.venv`/`node_modules`),
and prints the URL.

Public URL (VPN-only): **https://granite-speech-demo.vpc-int.res.ibm.com**

## Architecture

```
browser (VPN) ─HTTPS─▶ Route ─▶ frontend Svc:3000 ─proxy─▶ backend Svc:7860 ─HTTPS─▶ Granite LLM (GPU cluster)
WebRTC media ◀───────────── TURN relay (see below) ─────────────▶ browser
```

Only the frontend has a Route; the backend is internal. Backend stays
**replicas: 1** (in-memory WebRTC session state).

## Cluster-specific quirks handled here

- **Quota** (2 CPU / 4Gi/ns): builds use small requests and `deploy.sh` parks the
  Deployments at 0 replicas during builds.
- **Ephemeral storage** (LimitRange 4Gi/container): build pods raised to 12Gi and
  `SkipLayers`; the backend Dockerfile scopes `chmod` to the writable dirs only.
- **DNS/egress**: the cluster's internal resolver can't resolve public names, so
  the backend pod is pointed at public DNS (`dnsPolicy: None`, 1.1.1.1/8.8.8.8) —
  the LLM host, STUN, and TURN are all publicly resolvable. Egress itself works.
- **Missing libs**: backend image installs `libgl1`/`libglib2.0-0` (cv2) and
  `libgomp1` (onnxruntime), and pre-fetches the Kokoro model + NLTK `punkt_tab`.

## ⚠️ Remaining step: WebRTC TURN relay

Everything works except the audio media path. This cluster's egress NAT is
**symmetric**, so STUN-only peer-to-peer fails, and no free public TURN is usable
(fastrtc is SERVFAIL globally; openrelay is unreachable/deprecated). A reachable
**TURN relay** is required. Two options:

1. **Bring your own TURN** — set `ICE_SERVERS` in `config.env` (see
   `config.example.env`) to a Twilio/ExpressTURN/metered/corporate TURN, re-run
   `deploy.sh`. One config change, no new infra.
2. **Self-host coturn** — `openshift/coturn.yaml` deploys coturn behind a
   `LoadBalancer` (needs a VPN-reachable UDP LB — verify this cluster provides one):
   ```bash
   oc apply -f openshift/coturn.yaml
   IP=$(oc get svc coturn -o jsonpath='{.status.loadBalancer.ingress[0].ip}')   # or .hostname
   oc set env deploy/coturn EXTERNAL_IP="$IP"        # coturn must advertise the reachable IP
   # then set ICE_SERVERS to turn:$IP:3478 (user granite / cred granite) and re-run deploy.sh
   ```

The backend already reads `ICE_SERVERS` and hands the same servers to the browser
(via `/start`) and its own aiortc — so setting it fixes both ends.
