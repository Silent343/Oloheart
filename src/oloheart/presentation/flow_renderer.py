"""Three GPU draw calls for directional flow paths, arrows and moving markers."""

import ctypes
import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE, GL_TRUE,
    GL_LINES, GL_TRIANGLES, GL_POINTS, GL_PROGRAM_POINT_SIZE,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer,
    glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
    glUseProgram, glGetUniformLocation, glUniformMatrix4fv, glUniform1i,
    glUniform1f, glDrawArrays, glEnable, glDeleteBuffers, glDeleteVertexArrays, glDeleteProgram,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from oloheart.domain.model import anatomy_system_for
from oloheart.infrastructure.circulation_geometry import build_flow_routes, circulation_visible
from oloheart.presentation.scene_layout import placement_matrix

VERTEX = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
uniform mat4 mvp;
uniform float pixelRatio;
out vec3 tint;
void main() {
    gl_Position=mvp*vec4(position,1.0);
    gl_PointSize=6.0*pixelRatio;
    tint=color;
}
"""
FRAGMENT = """#version 330 core
in vec3 tint;
uniform int roundPoint;
out vec4 frag;
void main() {
    float alpha=1.0;
    if(roundPoint==1) {
        float d=length(gl_PointCoord-vec2(0.5))*2.0;
        if(d>1.0) discard;
        alpha=1.0-smoothstep(0.6,1.0,d);
    }
    frag=vec4(tint,alpha);
}
"""


class BloodFlowRenderer:
    """Render explanatory circulation in the same model coordinates as anatomy."""

    def __init__(self):
        self.routes = build_flow_routes()
        self.program = 0
        self._cached_geometry = {}
        self._particle_offsets = np.linspace(0,1,8,endpoint=False)
        for route in self.routes:
            color = np.asarray((1.0,.23,.22) if route.oxygenated else (.13,.67,1.0),dtype=np.float32)
            points = route.points
            ends = points[::4]
            line_positions = np.stack((ends[:-1],ends[1:]),axis=1).reshape(-1,3)
            line = np.column_stack((line_positions,np.tile(color*.34,(len(line_positions),1))))
            arrows = []
            for sample in (34,70):
                tip = points[sample]
                tangent = points[sample+1]-points[sample-1]
                tangent /= max(float(np.linalg.norm(tangent)),1e-5)
                side = np.cross(tangent,(0,0,1))
                if np.linalg.norm(side)<.1:
                    side = np.cross(tangent,(0,1,0))
                side /= max(float(np.linalg.norm(side)),1e-5)
                tail = tip-tangent*.055
                arrows.extend((tip,tail+side*.021,tail-side*.021))
            arrow = np.column_stack((arrows,np.tile(color*.85,(len(arrows),1))))
            dim = arrow.copy()
            dim[:,3:] *= .30
            self._cached_geometry[route.key] = (color,line.astype(np.float32),arrow.astype(np.float32),dim.astype(np.float32))

    def initialize(self):
        self.program = compileProgram(compileShader(VERTEX,GL_VERTEX_SHADER),
                                      compileShader(FRAGMENT,GL_FRAGMENT_SHADER))
        self.vao = int(glGenVertexArrays(1))
        self.vbo = int(glGenBuffers(1))
        self.uniforms = {k:glGetUniformLocation(self.program,k) for k in ("mvp","roundPoint","pixelRatio")}
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        for index,offset in ((0,0),(1,12)):
            glEnableVertexAttribArray(index)
            glVertexAttribPointer(index,3,GL_FLOAT,GL_FALSE,24,ctypes.c_void_p(offset))
        glBindVertexArray(0)

    def draw(self,state,projection,view,scale,pixel_ratio):
        if not circulation_visible(state):
            return
        lines,arrows,particles = [],[],[]
        for route in self.routes:
            if any(anatomy_system_for(part) not in state.visible_systems for part in route.parts):
                continue
            color,line,arrow,dim = self._cached_geometry[route.key]
            points = route.points
            lines.append(line)
            active = route.active(state)
            arrows.append(arrow if active else dim)
            if active:
                clock = state.flow_times[route.gate]
                phase = (clock*(.50 if route.gate == 0 else .85)) % 1.0
                samples = ((phase+self._particle_offsets)%1)*99
                indices = samples.astype(np.int32)
                weight = (samples-indices)[:,None]
                positions = points[indices]*(1-weight)+points[np.minimum(indices+1,99)]*weight
                particles.append(np.column_stack((positions,np.tile(color,(8,1)))))
        model = placement_matrix(state,(0,0,0),self.routes[0].parts[0],scale)
        glUseProgram(self.program)
        glUniformMatrix4fv(self.uniforms["mvp"],1,GL_TRUE,projection@view@model)
        glUniform1f(self.uniforms["pixelRatio"],pixel_ratio)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        for primitive,data,round_point in ((GL_LINES,lines,0),(GL_TRIANGLES,arrows,0),(GL_POINTS,particles,1)):
            if not data:
                continue
            array = np.concatenate(data).astype(np.float32,copy=False)
            glBufferData(GL_ARRAY_BUFFER,array.nbytes,array,GL_DYNAMIC_DRAW)
            glUniform1i(self.uniforms["roundPoint"],round_point)
            glDrawArrays(primitive,0,len(array))
        glBindVertexArray(0)

    def dispose(self):
        if self.program:
            glDeleteBuffers(1,[self.vbo])
            glDeleteVertexArrays(1,[self.vao])
            glDeleteProgram(self.program)
            self.program = 0
