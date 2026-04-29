import os
import pickle
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, zoom
from skimage import measure


# ── Graph helpers ────────────────────────────────────────────────

KERNEL_26 = np.ones((3, 3, 3), dtype=np.uint8)
KERNEL_26[1, 1, 1] = 0


def classify_voxels_fast(skel):
    skel_u8 = skel.astype(np.uint8)
    ncount  = ndimage.convolve(skel_u8, KERNEL_26, mode='constant', cval=0) * skel_u8
    return (list(map(tuple, np.argwhere(ncount == 1))),
            list(map(tuple, np.argwhere(ncount >= 3))),
            ncount)


def cluster_voxels(voxels, radius):
    if not voxels:
        return []
    arr  = np.array(voxels, dtype=float)
    used = np.zeros(len(arr), dtype=bool)
    clusters = []
    for i in range(len(arr)):
        if used[i]:
            continue
        dists = np.linalg.norm(arr - arr[i], axis=1)
        idx   = np.where((dists <= radius) & ~used)[0]
        used[idx] = True
        centroid = arr[idx].mean(axis=0)
        closest  = idx[np.argmin(np.linalg.norm(arr[idx] - centroid, axis=1))]
        clusters.append((tuple(arr[closest].astype(int)),
                         [tuple(arr[j].astype(int)) for j in idx]))
    return clusters


def get_26_neighbors(z, y, x, shape):
    Z, Y, X = shape
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                    yield (nz, ny, nx)


def build_clean_skeleton_and_nodes(skel, bif_clusters, ep_clusters):
    node_set   = set()
    raw_to_rep = {}
    for rep, members in bif_clusters + ep_clusters:
        node_set.add(rep)
        for m in members:
            raw_to_rep[m] = rep
    skel_set = set(map(tuple, np.argwhere(skel)))
    return skel_set, node_set, raw_to_rep


def trace_segment(start, first_step, skel_set, node_set, raw_to_rep, shape,
                  max_steps=50000):
    path    = [start, first_step]
    visited = {start, first_step}
    prev, curr = start, first_step
    for _ in range(max_steps):
        if curr in raw_to_rep:
            rep = raw_to_rep[curr]
            if rep != start and rep in node_set:
                path[-1] = rep
                return rep, path
        if curr in node_set and curr != start:
            return curr, path
        nbrs = [nb for nb in get_26_neighbors(*curr, shape)
                if nb in skel_set and nb not in visited]
        if not nbrs:
            return None, path
        if len(nbrs) == 1:
            next_v = nbrs[0]
        else:
            direction = np.array(curr) - np.array(prev)
            next_v = max(nbrs, key=lambda nb: np.dot(np.array(nb) - np.array(curr), direction))
        visited.add(next_v)
        prev, curr = curr, next_v
        path.append(curr)
    return None, path


def arc_length_mm(path, voxel_size):
    if len(path) < 2:
        return 0.0
    p = np.array(path, dtype=float) * np.array(voxel_size)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def extract_graph(skel, bif_clusters, ep_clusters, voxel_size):
    bifs  = [rep for rep, _ in bif_clusters]
    eps   = [rep for rep, _ in ep_clusters]
    shape = skel.shape
    skel_set, node_set, raw_to_rep = build_clean_skeleton_and_nodes(
        skel, bif_clusters, ep_clusters)
    node_ids = {v: f"BIF_{i:02d}" for i, v in enumerate(bifs)}
    node_ids.update({v: f"EP_{i:02d}" for i, v in enumerate(eps)})

    edges = {}
    for start in node_set:
        for first_step in get_26_neighbors(*start, shape):
            if first_step not in skel_set:
                continue
            if first_step in raw_to_rep and raw_to_rep[first_step] == start:
                continue
            end, path = trace_segment(start, first_step, skel_set, node_set,
                                      raw_to_rep, shape)
            if end is None or end == start:
                continue
            key     = frozenset([start, end])
            new_len = arc_length_mm(path, voxel_size)
            if key in edges and new_len >= edges[key]['length_mm']:
                continue
            edges[key] = {
                'path':     path,
                'node_ids': (node_ids.get(start, str(start)),
                             node_ids.get(end,   str(end))),
                'length_mm': new_len,
            }
    return node_ids, edges


# ── Surface mesh + HTML ──────────────────────────────────────────

def voxel_to_mm(vox, affine):
    return (affine @ np.array([vox[0], vox[1], vox[2], 1.0]))[:3]


def extract_surface_mesh(mask, affine, downsample=4):
    small = zoom(mask.astype(float), 1.0 / downsample, order=1) > 0.5 if downsample > 1 else mask
    if not small.sum():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    verts, faces, _, _ = measure.marching_cubes(small, level=0.5)
    verts_orig = verts * downsample
    verts_mm   = (affine @ np.hstack([verts_orig, np.ones((len(verts_orig), 1))]).T).T[:, :3]
    return verts_mm, faces


def generate_html(verts, faces, nodes, edge_paths, out_path):
    import json as _json
    center     = verts.mean(axis=0).tolist() if len(verts) else [0, 0, 0]
    mesh_data  = {'vertices': verts.tolist(), 'faces': faces.tolist(), 'center': center}
    nodes_data = [{'pos': n['pos'], 'label': n['label'], 'type': n['type']} for n in nodes]
    edges_data = [{'points': e['points'], 'label': e['label'], 'color': e['color']}
                  for e in edge_paths]
    PAL = ['#ff6b6b','#4ecdc4','#45b7d1','#96ceb4','#ffeaa7','#dfe6e9','#fd79a8','#a29bfe',
           '#00b894','#e17055','#74b9ff','#55efc4','#fab1a0','#81ecec','#b2bec3','#6c5ce7',
           '#fdcb6e','#e84393','#00cec9','#ff7675']
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Vessel Graph 3D</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{background:#0a0a0f;overflow:hidden;font-family:monospace}}canvas{{display:block}}#info{{position:absolute;top:16px;left:16px;color:#8899aa;font-size:12px;line-height:1.6;background:rgba(10,10,15,.85);padding:12px 16px;border:1px solid #223;border-radius:6px;max-height:90vh;overflow-y:auto}}#info h2{{color:#ddeeff;font-size:14px;margin-bottom:8px}}#controls{{position:absolute;bottom:16px;left:16px;color:#667;font-size:11px;background:rgba(10,10,15,.7);padding:8px 12px;border-radius:4px}}.sr{{display:flex;align-items:center;gap:8px;margin:6px 0;color:#aab}}.sr input[type=range]{{width:120px}}.sr label{{font-size:11px;min-width:80px}}.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;flex-shrink:0}}.li{{display:flex;align-items:center;gap:8px;margin:3px 0}}</style></head>
<body><div id="info"><h2>Vessel Graph 3D</h2>
<div class="sr"><label>Vessel opacity</label><input type="range" id="opSlider" min="0" max="100" value="15"><span id="opVal">0.15</span></div>
<div class="sr"><label>Edge width</label><input type="range" id="wSlider" min="1" max="10" value="3"><span id="wVal">3</span></div>
<div class="sr"><label>Node size</label><input type="range" id="nSlider" min="1" max="20" value="8"><span id="nVal">0.8</span></div>
<div class="sr"><label>Labels</label><input type="checkbox" id="lblToggle" checked></div>
<div id="legend"></div></div>
<div id="controls">Drag=rotate · Scroll=zoom · Right-drag=pan</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const MESH={json.dumps(mesh_data)};const NODES={json.dumps(nodes_data)};const EDGES={json.dumps(edges_data)};
const PAL={json.dumps(PAL)};
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0a0a0f);
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,.1,5000);
const renderer=new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0x404060,.6));
const d1=new THREE.DirectionalLight(0xffffff,.8);d1.position.set(1,1,1);scene.add(d1);
const cx=MESH.center[0],cy=MESH.center[1],cz=MESH.center[2];
let vesselMesh=null;
if(MESH.vertices.length>0){{const g=new THREE.BufferGeometry();const v=new Float32Array(MESH.vertices.length*3);MESH.vertices.forEach((p,i)=>{{v[i*3]=p[0]-cx;v[i*3+1]=p[1]-cy;v[i*3+2]=p[2]-cz;}});g.setAttribute('position',new THREE.BufferAttribute(v,3));g.setIndex(MESH.faces.flat());g.computeVertexNormals();vesselMesh=new THREE.Mesh(g,new THREE.MeshPhongMaterial({{color:0x6688bb,transparent:true,opacity:.15,side:THREE.DoubleSide,depthWrite:false}}));scene.add(vesselMesh);}}
const edgeMats=[];EDGES.forEach((e,i)=>{{const pts=e.points.map(p=>new THREE.Vector3(p[0]-cx,p[1]-cy,p[2]-cz));const m=new THREE.LineBasicMaterial({{color:e.color||PAL[i%PAL.length],linewidth:3}});scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),m));edgeMats.push(m);}});
const ng=new THREE.Group();const lbls=[];NODES.forEach(n=>{{const c=n.type==='BIF'?0xff3333:0x3399ff;const m=new THREE.Mesh(new THREE.SphereGeometry(.8,16,16),new THREE.MeshPhongMaterial({{color:c,emissive:c,emissiveIntensity:.3}}));m.position.set(n.pos[0]-cx,n.pos[1]-cy,n.pos[2]-cz);ng.add(m);const cv=document.createElement('canvas');cv.width=256;cv.height=64;const ctx=cv.getContext('2d');ctx.font='bold 36px monospace';ctx.fillStyle=n.type==='BIF'?'#ff6666':'#66aaff';ctx.textAlign='center';ctx.fillText(n.label,128,44);const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:new THREE.CanvasTexture(cv),transparent:true,depthTest:false}}));sp.position.set(n.pos[0]-cx,n.pos[1]-cy+2,n.pos[2]-cz);sp.scale.set(8,2,1);ng.add(sp);lbls.push(sp);}});scene.add(ng);
let mn=new THREE.Vector3(Infinity,Infinity,Infinity),mx=new THREE.Vector3(-Infinity,-Infinity,-Infinity);NODES.forEach(n=>{{mn.x=Math.min(mn.x,n.pos[0]-cx);mn.y=Math.min(mn.y,n.pos[1]-cy);mn.z=Math.min(mn.z,n.pos[2]-cz);mx.x=Math.max(mx.x,n.pos[0]-cx);mx.y=Math.max(mx.y,n.pos[1]-cy);mx.z=Math.max(mx.z,n.pos[2]-cz);}});
const ext=mx.clone().sub(mn).length();let sph={{theta:0,phi:Math.PI/2,radius:ext*1.5}};let pan=new THREE.Vector3();let drag=false,rdrag=false,pm={{x:0,y:0}};
function uc(){{const r=sph.radius,p=sph.phi,t=sph.theta;camera.position.set(r*Math.sin(p)*Math.sin(t)+pan.x,r*Math.cos(p)+pan.y,r*Math.sin(p)*Math.cos(t)+pan.z);camera.lookAt(pan);}}uc();
renderer.domElement.addEventListener('mousedown',e=>{{if(e.button===2)rdrag=true;else drag=true;pm={{x:e.clientX,y:e.clientY}};}});
renderer.domElement.addEventListener('mousemove',e=>{{const dx=e.clientX-pm.x,dy=e.clientY-pm.y;if(drag){{sph.theta-=dx*.005;sph.phi=Math.max(.1,Math.min(Math.PI-.1,sph.phi-dy*.005));uc();}}if(rdrag){{const r=new THREE.Vector3().crossVectors(camera.up,camera.getWorldDirection(new THREE.Vector3())).normalize();pan.addScaledVector(r,dx*sph.radius*.001);pan.addScaledVector(camera.up,-dy*sph.radius*.001);uc();}}pm={{x:e.clientX,y:e.clientY}};}});
renderer.domElement.addEventListener('mouseup',()=>{{drag=false;rdrag=false;}});
renderer.domElement.addEventListener('wheel',e=>{{sph.radius=Math.max(5,Math.min(ext*5,sph.radius*(1+e.deltaY*.001)));uc();}});
renderer.domElement.addEventListener('contextmenu',e=>e.preventDefault());
document.getElementById('opSlider').addEventListener('input',e=>{{const v=e.target.value/100;document.getElementById('opVal').textContent=v.toFixed(2);if(vesselMesh)vesselMesh.material.opacity=v;}});
document.getElementById('wSlider').addEventListener('input',e=>{{const v=parseInt(e.target.value);document.getElementById('wVal').textContent=v;edgeMats.forEach(m=>m.linewidth=v);}});
document.getElementById('nSlider').addEventListener('input',e=>{{const v=parseInt(e.target.value)/10;document.getElementById('nVal').textContent=v.toFixed(1);ng.children.forEach(c=>{{if(c.isMesh)c.scale.setScalar(v);}});}});
document.getElementById('lblToggle').addEventListener('change',e=>{{lbls.forEach(s=>s.visible=e.target.checked);}});
let lh='<div class="li"><span class="dot" style="background:#ff3333"></span>Bifurcation</div><div class="li"><span class="dot" style="background:#3399ff"></span>Endpoint</div><hr style="border-color:#223;margin:6px 0">';
EDGES.forEach((e,i)=>{{lh+=`<div class="li"><span class="dot" style="background:${{e.color||PAL[i%PAL.length]}}"></span>${{e.label}}</div>`;}});
document.getElementById('legend').innerHTML=lh;
(function animate(){{requestAnimationFrame(animate);renderer.render(scene,camera);}})();
window.addEventListener('resize',()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);}});
</script></body></html>"""
    with open(out_path, 'w') as f:
        f.write(html)


# ── Main function ────────────────────────────────────────────────

def build_vessel_graph(planb_output_dir, planb_data_dir, out_dir,
                       skip_segments=None, n_phases=25,
                       bif_cluster_radius=3, ep_cluster_radius=2,
                       mask_downsample=2):
    """Build reference vessel graph from Plan B+ segments.

    Args:
        planb_output_dir:  Part B output directory (segments.json, p01_binary.nii.gz, ...)
        planb_data_dir:    directory with cropped NIfTI phase files
        out_dir:           output directory for graph, PKL files, HTML
        skip_segments:     list of segment IDs to exclude
        n_phases:          number of cardiac phases
        bif_cluster_radius, ep_cluster_radius: clustering radii in voxels
        mask_downsample:   surface mesh downsampling factor (2=detailed, 4=fast)
    """
    if skip_segments is None:
        skip_segments = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 9: Build vessel graph")
    print(f"  planb_output_dir: {planb_output_dir}")
    print(f"  out_dir         : {out_dir}")
    print("=" * 60)

    with open(os.path.join(planb_output_dir, "segments.json")) as f:
        seg_data = json.load(f)
    all_segments = {int(sid): np.array(coords) for sid, coords in seg_data.items()}
    segments = {sid: c for sid, c in all_segments.items() if sid not in skip_segments}
    print(f"  Active segments: {len(segments)} / {len(all_segments)}")

    p1_img     = nib.load(os.path.join(planb_output_dir, "p01_binary.nii.gz"))
    mask_shape = p1_img.shape
    affine     = p1_img.affine
    vox_size   = np.abs(np.diag(affine)[:3])
    print(f"  Shape: {mask_shape}  vox: {vox_size}")

    skel = np.zeros(mask_shape, dtype=np.uint8)
    for sid, coords in segments.items():
        skel[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    eps_raw, bifs_raw, _ = classify_voxels_fast(skel)
    bif_clusters = cluster_voxels(bifs_raw, bif_cluster_radius)
    ep_clusters  = cluster_voxels(eps_raw,  ep_cluster_radius)
    bifs = [r for r, _ in bif_clusters]
    eps  = [r for r, _ in ep_clusters]
    print(f"  Graph: bif={len(bifs)}  ep={len(eps)}")

    node_ids, edges = extract_graph(skel, bif_clusters, ep_clusters, vox_size)
    print(f"  Edges: {len(edges)}")

    ref_data = {
        'source': 'planb_segments', 'phase': 1,
        'affine': affine, 'voxel_size': vox_size, 'shape': mask_shape,
        'skel': skel, 'segments': segments,
        'bifs': bifs, 'eps': eps, 'node_ids': node_ids, 'edges': edges,
        'skip_segments': skip_segments,
    }
    with open(os.path.join(out_dir, "reference_graph.pkl"), 'wb') as f:
        pickle.dump(ref_data, f)

    overlay = np.zeros(mask_shape, dtype=np.uint8)
    overlay[skel > 0] = 1
    for group, label in [(bifs, 2), (eps, 3)]:
        for v in group:
            z0, y0, x0 = int(v[0]), int(v[1]), int(v[2])
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dz*dz + dy*dy + dx*dx <= 4:
                            zz, yy, xx = z0+dz, y0+dy, x0+dx
                            if (0 <= zz < mask_shape[0] and
                                0 <= yy < mask_shape[1] and
                                0 <= xx < mask_shape[2]):
                                overlay[zz, yy, xx] = label
    nib.save(nib.Nifti1Image(overlay, affine),
             os.path.join(out_dir, "reference_skel_overlay.nii.gz"))

    skel_colored = np.zeros(mask_shape, dtype=np.uint8)
    for sid, coords in segments.items():
        skel_colored[coords[:, 0], coords[:, 1], coords[:, 2]] = sid
    nib.save(nib.Nifti1Image(skel_colored, affine),
             os.path.join(out_dir, "reference_segments.nii.gz"))

    for ph in range(1, n_phases + 1):
        fname = os.path.join(planb_output_dir, f"p{ph:02d}_binary.nii.gz")
        if not os.path.exists(fname):
            continue
        img_ph  = nib.load(fname)
        mask_ph = img_ph.get_fdata().astype(bool)
        out = os.path.join(out_dir, f"phase_{ph:02d}_data.pkl")
        with open(out, 'wb') as f:
            pickle.dump({'phase': ph, 'affine': img_ph.affine,
                         'voxel_size': np.abs(np.diag(img_ph.affine)[:3]),
                         'shape': mask_ph.shape, 'mask': mask_ph}, f)
        print(f"  p{ph:02d}: fg={mask_ph.sum()} -> {out}")

    print("\nGenerating 3D HTML visualization...")
    nids     = {tuple(int(x) for x in k): v for k, v in node_ids.items()}
    mask_p1  = p1_img.get_fdata().astype(bool)
    verts, faces = extract_surface_mesh(mask_p1, affine, downsample=mask_downsample)

    PAL = ['#ff6b6b','#4ecdc4','#45b7d1','#96ceb4','#ffeaa7','#dfe6e9','#fd79a8','#a29bfe',
           '#00b894','#e17055','#74b9ff','#55efc4','#fab1a0','#81ecec','#b2bec3','#6c5ce7',
           '#fdcb6e','#e84393','#00cec9','#ff7675']
    nodes_vis = [{'pos': voxel_to_mm(vox, affine).tolist(), 'label': label,
                  'type': 'BIF' if label.startswith('BIF') else 'EP'}
                 for vox, label in nids.items()]
    edge_paths = []
    for i, (key, e) in enumerate(sorted(edges.items(), key=lambda x: x[1]['node_ids'])):
        path   = e['path']
        step   = max(1, len(path) // 100)
        sampled = path[::step]
        if path[-1] not in sampled:
            sampled.append(path[-1])
        edge_paths.append({
            'points': [voxel_to_mm(v, affine).tolist() for v in sampled],
            'label':  f"{e['node_ids'][0]} — {e['node_ids'][1]} ({e['length_mm']:.1f}mm)",
            'color':  PAL[i % len(PAL)],
        })

    html_path = os.path.join(out_dir, "vessel_graph_3d.html")
    generate_html(verts, faces, nodes_vis, edge_paths, html_path)
    print(f"  HTML -> {html_path}")
    print(f"\nDone  ->  {out_dir}")
