use anyhow::Result;
use std::sync::Arc;
use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

mod renderer;
use renderer::{Movement, Renderer};

fn main() -> Result<()> {
    pollster::block_on(run())
}

async fn run() -> Result<()> {
    let event_loop = EventLoop::new()?;
    // Wrap the window in Arc so we can share it with the renderer and the closure.
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("PhotonForge RT — Rust/wgpu")
            .build(&event_loop)?,
    );

    // Renderer needs a &Window
    let mut renderer = Renderer::new(window.as_ref()).await?;

    let mut mouse_down = false;
    let mut last_mouse_pos: Option<(f32, f32)> = None;
    let win_for_loop = window.clone(); // capture a clone for the event loop

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(size) => renderer.resize(size),
                WindowEvent::RedrawRequested => {
                    if let Err(e) = renderer.render() {
                        eprintln!("render error: {e:?}");
                    }
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    handle_keyboard(&mut renderer, &key_event);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        mouse_down = state == ElementState::Pressed;
                        if !mouse_down {
                            last_mouse_pos = None;
                        }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if mouse_down {
                        if let Some((lx, ly)) = last_mouse_pos {
                            let dx = position.x as f32 - lx;
                            let dy = position.y as f32 - ly;
                            renderer.on_mouse_delta(dx, dy);
                        }
                        last_mouse_pos = Some((position.x as f32, position.y as f32));
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let s = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32,
                    };
                    renderer.on_scroll(s);
                }
                _ => {}
            },
            Event::AboutToWait => {
                // Request redraw each tick on the Arc<Window>
                win_for_loop.request_redraw();
            }
            _ => {}
        }
    })?; // propagate the Result from run()

    Ok(())
}

fn handle_keyboard(renderer: &mut Renderer, key_event: &KeyEvent) {
    if key_event.state != ElementState::Pressed {
        return;
    }
    match key_event.logical_key.clone() {
        Key::Named(NamedKey::Escape) => std::process::exit(0),
        Key::Named(NamedKey::Space) => renderer.reset_accum(),
        Key::Character(txt) => match txt.as_str() {
            "w" | "W" => renderer.queue_movement(Movement::Forward),
            "s" | "S" => renderer.queue_movement(Movement::Backward),
            "a" | "A" => renderer.queue_movement(Movement::Left),
            "d" | "D" => renderer.queue_movement(Movement::Right),
            "q" | "Q" => renderer.queue_movement(Movement::Down),
            "e" | "E" => renderer.queue_movement(Movement::Up),
            "r" | "R" => renderer.reset_accum(),
            _ => {}
        },
        _ => {}
    }
}
