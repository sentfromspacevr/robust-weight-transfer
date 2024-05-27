import numpy as np
import scipy as sp
import bpy


def get_obj_arrs_world(object: bpy.types.Object):
    mesh: bpy.types.Mesh = object.data
    mesh.calc_loop_triangles()

    vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    indices = np.empty((len(mesh.loop_triangles), 3), dtype=np.int64)
    normals = np.empty((len(mesh.vertices), 3), dtype=np.float32)

    mesh.vertices.foreach_get("co", vertices.reshape(-1))
    mesh.loop_triangles.foreach_get("vertices", indices.reshape(-1))
    # mesh.vertex_normals.foreach_get('vector', normals.reshape(-1))
    mesh.vertices.foreach_get('normal', normals.reshape(-1))
    
    world_matrix = np.array(object.matrix_world)
    ones = np.ones((vertices.shape[0], 1))
    
    vertices_4d = np.hstack((vertices, ones))
    world_vertices_4d = (world_matrix @ vertices_4d.T).T
    world_vertices_4d = np.ascontiguousarray(world_vertices_4d, dtype=np.float32)
    world_vertices = world_vertices_4d[:,:3] / world_vertices_4d[:, 3][:, np.newaxis]
    
    world_normals = (np.linalg.inv(world_matrix[:3, :3]).T @ normals.T).T
    world_normals = np.ascontiguousarray(world_normals, dtype=np.float32)
    
    return world_vertices, indices, world_normals


def get_group_arr(obj: bpy.types.Object, group_name):
    mesh: bpy.types.Mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh): return
    group_index = obj.vertex_groups[group_name].index
    arr = np.zeros(len(mesh.vertices), dtype=np.float32)
    for i, v in enumerate(mesh.vertices):
        for g in v.groups:
            if g.group == group_index:
                arr[i] = g.weight
    return arr


def get_groups_arr(obj: bpy.types.Object, include_groups: list[bool]=None):
    mesh: bpy.types.Mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh): return

    arr = np.zeros((len(mesh.vertices), len(obj.vertex_groups)), dtype=np.float32)
    for i, v in enumerate(mesh.vertices):
        current_vertex = arr[i]
        for g in v.groups:
            if include_groups and include_groups[g.group]:
                current_vertex[g.group] = g.weight
            elif not include_groups:
                current_vertex[g.group] = g.weight
    return arr


def draw_debug_vertex_colors(obj, matched):
    mesh: bpy.types.Mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh): return

    if "RBT Matched" in mesh.vertex_colors:
        color_layer = mesh.vertex_colors["RBT Matched"]
    else:
        color_layer = mesh.vertex_colors.new(name="RBT Matched")
    if not color_layer: return False
    color_layer.active = True
    loop_ind = np.zeros(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get('vertex_index', loop_ind)
    loop_matched = matched[loop_ind]
    color_data = np.ones((len(mesh.loops), 4), dtype=np.float32)
    color_data[~loop_matched] = [234/255, 0, 255/255, 1.0]
    color_layer.data.foreach_set("color", color_data.reshape(-1))
    mesh.update()
    mesh.vertex_colors.active = color_layer
    return True
    
    
TOPOLOGY_MODS = {
    'ARRAY',
    'BEVEL',
    'BOOLEAN',
    'BUILD',
    'DECIMATE',
    'EDGE_SPLIT',
    'MASK',
    'MIRROR',
    'MULTIRES',
    'REMESH',
    'SCREW',
    'SKIN',
    'SOLIDIFY',
    'SUBSURF',
    'TRIANGULATE',
    'WELD',
    'WIREFRAME'
}


def has_modifier(obj: bpy.types.Object, *mod_types):
    if obj and obj.type == 'MESH' and obj.modifiers:
        return any(mod.type in mod_types for mod in obj.modifiers)
    return False


# TODO: source object required to have armature modifier
# throw exceptions
def is_vertex_group_deform_bone(obj, group_name):
    armature_mod = None
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE':
            armature_mod = mod
            break

    if not armature_mod or not armature_mod.object or armature_mod.object.type != 'ARMATURE':
        return False

    armature_obj = armature_mod.object
    bone = armature_obj.data.bones.get(group_name)

    if bone and bone.use_deform:
        return True

    return False


def get_mesh_adjacency_matrix_sparse(mesh: bpy.types.Mesh, include_self=False):
    edge_data = np.empty((len(mesh.edges), 2), dtype=int)
    mesh.edges.foreach_get("vertices", edge_data.reshape(-1))
    num_verts = len(mesh.vertices)
    rows = np.hstack([edge_data[:, 0], edge_data[:, 1]])
    cols = np.hstack([edge_data[:, 1], edge_data[:, 0]])
    data = np.ones(len(rows), dtype=int)  # Corresponding data entries for the CSR matrix

    # Create a symmetric adjacency matrix (since each edge is undirected)
    adjacency_matrix = sp.sparse.csr_array((data, (rows, cols)), shape=(num_verts, num_verts))
    if include_self:
        adjacency_matrix.setdiag(1)
    return adjacency_matrix
    
    
def get_mesh_adjacency_list(mesh: bpy.types.Mesh):
    edge_data = np.empty((len(mesh.edges), 2), dtype=int)
    mesh.edges.foreach_get("vertices", edge_data.reshape(-1))
    num_verts = len(mesh.vertices)
    adj_list = [[] for _ in range(num_verts)]
    for edge in edge_data:
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])
    return adj_list


def write_weights(obj, weights, names, threshold=0):
    groups = obj.vertex_groups
    for name, w in zip(names, weights.T):
        created = False
        if name not in groups:
            created = True
            obj.vertex_groups.new(name=name)

        group = obj.vertex_groups[name]
        if group.lock_weight:
            continue
            
        for j, wv in enumerate(w):
            if wv >= threshold:
                group.add([j], wv, 'REPLACE')
        if not created:
            ind = np.where(w < threshold)[0].tolist()
            group.remove(ind)
            
            
def is_group_valid(vertex_groups, group_name):
    return len(group_name) > 0 and group_name in vertex_groups