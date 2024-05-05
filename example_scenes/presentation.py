from manim import *
from manim.opengl import *  # type: ignore
import manim.utils.opengl as opengl
from pyglet.window import key

config["background_color"] = "#222"
config["fullscreen"] = True
config["frame_width"] = 16
config["frame_height"] = 9

class Intro(Scene):
    def construct(self):
        win = self.renderer.window._window
        # win.set_fullscreen(True)

        key_text = Text("")

        @win.event
        def on_key_press(symbol, modifiers):
            # self.play(Transform(key_text,Text(str(symbol))))
            print(symbol)
            # if symbol == key.ESCAPE:
            #     win.close()

        current_scene = 0
        self.next_section("Intro", skip_animations=current_scene!=0)
        self.wait()
        text = Text("The Verts and Frags of OpenGL")
        self.play(FadeIn(text), shift=UP)

        self.interactive_embed()
        self.interactive_embed()
        self.interactive_embed()
        self.interactive_embed()
        self.interactive_embed()
        self.interactive_embed()
        self.interactive_embed()
        self.next_section("Circle", skip_animations=current_scene!=1)

        c = Circle(fill_opacity=0).shift(UL)
        self.add(c)

        shader = Shader(
            self.renderer.context,
            source={
                "vertex_shader": """
                #version 330

                in vec4 in_vert;
                in vec4 in_color;
                out vec4 v_color;
                uniform mat4 u_model_view_matrix;
                uniform mat4 u_projection_matrix;

                void main() {
                    v_color = in_color;
                    vec4 camera_space_vertex = u_model_view_matrix * in_vert;
                    vec4 clip_space_vertex = u_projection_matrix * camera_space_vertex;
                    gl_Position = clip_space_vertex;
                }
            """,
                "fragment_shader": """
            #version 330

            in vec4 v_color;
            out vec4 frag_color;

            void main() {
              frag_color = v_color;
            }
            """,
            },
        )
        shader.set_uniform("u_model_view_matrix", opengl.view_matrix())
        shader.set_uniform(
            "u_projection_matrix",
            opengl.orthographic_projection_matrix(),
        )

        attributes = np.zeros(
            6,
            dtype=[
                ("in_vert", np.float32, (4,)),
                ("in_color", np.float32, (4,)),
            ],
        )
        attributes["in_vert"] = np.array(
            [
                [-1, -1, 0, 1],
                [-1, 1, 0, 1],
                [1, 1, 0, 1],
                [-1, -1, 0, 1],
                [1, -1, 0, 1],
                [1, 1, 0, 1],
            ],
        )
        attributes["in_color"] = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
        )
        mesh = Mesh(shader, attributes)
        self.add(mesh)

        self.wait(5)

if __name__ == "__main__":
    with tempconfig({"renderer":"opengl", "preview":True, "force_window":True, "write_to_movie":False}):
        Intro().render()
