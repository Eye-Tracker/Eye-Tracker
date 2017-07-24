use piston_window::{PistonWindow, UpdateEvent, Window, WindowSettings};
use piston_window::{G2d, G2dTexture, TextureSettings};
use piston_window::OpenGL;
use piston_window::texture::UpdateTexture;
use piston_window;
use conrod::{widget, Labelable, Positionable, Sizeable, Widget};
use conrod;
use std;
use find_folder;
use std::sync::{Arc, Mutex};
use std::cmp;

pub fn theme() -> conrod::Theme {
    use conrod::position::{Align, Direction, Padding, Position, Relative};
    conrod::Theme {
        name: "Eye Tracking Theme".to_string(),
        padding: Padding::none(),
        x_position: Position::Relative(Relative::Align(Align::Start), None),
        y_position: Position::Relative(Relative::Direction(Direction::Backwards, 20.0), None),
        background_color: conrod::color::DARK_CHARCOAL,
        shape_color: conrod::color::LIGHT_CHARCOAL,
        border_color: conrod::color::BLACK,
        border_width: 0.0,
        label_color: conrod::color::WHITE,
        font_id: None,
        font_size_large: 26,
        font_size_medium: 18,
        font_size_small: 12,
        widget_styling: conrod::theme::StyleMap::default(),
        mouse_drag_threshold: 0.0,
        double_click_threshold: std::time::Duration::from_millis(500),
    }
}

pub struct App {
    pub low_threshold: f32,
    pub high_threshold: f32,
    pub size_filter: f64,
    pub eye_pos: (u32, u32),
    pub eye_pos_threshold: u32,
    pub eye_min_radius: f32,
    pub eye_max_radius: f32,
}

impl App {
    pub fn new(low: f32, high: f32, size: f64, eye_pos: (u32, u32), eye_thresh: u32, min_radius: f32, max_radius: f32) -> Self {
        App {
            low_threshold: low, 
            high_threshold: high, 
            size_filter: size,
            eye_pos: eye_pos,
            eye_pos_threshold: eye_thresh,
            eye_min_radius: min_radius,
            eye_max_radius: max_radius
        }
    }
}

widget_ids! {
    pub struct Ids {
        canvas,
        title,
        low_threshold_slider,
        high_threshold_slider,
        size_filter_slider,
        eye_pos_xy,
        eye_pos_threshold_slider,
        eye_max_radius_slider,
        eye_min_radius_slider,
    }
}

fn gui(ui: &mut conrod::UiCell, ids: &Ids, dim_img: (u32, u32), app: Arc<Mutex<App>>) {
    const MARGIN: conrod::Scalar = 30.0;
    const TITLE_SIZE: conrod::FontSize = 42;

    const TITLE: &'static str = "Configuration";
    const LOW_THRESHOLD: &'static str = "Canny Edge Low Theshold:";
    const HIGH_THRESHOLD: &'static str = "Canny Edge High Theshold:";
    const CONTOUR_SIZE_FILTER: &'static str = "Countour Size Filtering:";
    const EYE_POS_XY: &'static str = "Eye position:";
    const EYE_POS_THRESHOLD: &'static str = "Eye Position Threshold:";
    const EYE_MAX_RADIUS: &'static str = "Maximum Eye Size radius:";
    const EYE_MIN_RADIUS: &'static str = "Minimum Eye Size radius:";

    widget::Canvas::new()
        .pad(MARGIN)
        .set(ids.canvas, ui);

    widget::Text::new(TITLE)
        .font_size(TITLE_SIZE)
        .mid_top_of(ids.canvas)
        .set(ids.title, ui);

    let mut unlocked = app.lock().unwrap();

    let low_label = format!("{} {}", LOW_THRESHOLD, unlocked.low_threshold as i16);
    if let Some(low_thres) = widget::Slider::new(unlocked.low_threshold, 0.0, 255.0)
        .w_h(275.0, 50.0)
        .mid_left_of(ids.canvas)
        .down_from(ids.title, 15.0)
        .label(&low_label)
        .set(ids.low_threshold_slider, ui) {
            unlocked.low_threshold = low_thres;
        };

    let high_label = format!("{} {}", HIGH_THRESHOLD, unlocked.high_threshold as i16);
    if let Some(high_thres) = widget::Slider::new(unlocked.high_threshold, 0.0, 255.0)
        .w_h(275.0, 50.0)
        .mid_left_of(ids.canvas)
        .down_from(ids.low_threshold_slider, 15.0)
        .label(&high_label)
        .set(ids.high_threshold_slider, ui) {
            unlocked.high_threshold = high_thres;
        };
    
    let size_label = format!("{} {}", CONTOUR_SIZE_FILTER, unlocked.size_filter as f32);
    if let Some(size) = widget::Slider::new(unlocked.size_filter * 6f64 * 100f64, 0.0, 100.0)
        .w_h(275.0, 50.0)
        .mid_left_of(ids.canvas)
        .down_from(ids.high_threshold_slider, 15.0)
        .label(&size_label)
        .set(ids.size_filter_slider, ui) {
            unlocked.size_filter = size / 6f64 / 100f64;
        };

    let size_label = format!("{} {}", CONTOUR_SIZE_FILTER, unlocked.size_filter as f32);
    if let Some(size) = widget::Slider::new(unlocked.size_filter * 100f64, 0.0, 100.0)
        .w_h(275.0, 50.0)
        .mid_left_of(ids.canvas)
        .down_from(ids.high_threshold_slider, 15.0)
        .label(&size_label)
        .set(ids.size_filter_slider, ui) {
            unlocked.size_filter = (size - 80f64) / 100f64;
        };

    let xy_label = format!("{} {},{}", EYE_POS_XY, unlocked.eye_pos.0, unlocked.eye_pos.1);
    for (x, y) in widget::XYPad::new(unlocked.eye_pos.0 as f32, 0f32, dim_img.0 as f32,
                                     unlocked.eye_pos.1 as f32, 0f32, dim_img.1 as f32)
        .label(&xy_label)
        .w_h(275.0, 275.0)
        .mid_left_of(ids.canvas)
        .down_from(ids.size_filter_slider, 15.0)
        .parent(ids.canvas)
        .set(ids.eye_pos_xy, ui)
    {
        unlocked.eye_pos = (x as u32, y as u32);
    }

    let eye_threshold_label = format!("{} {}", EYE_POS_THRESHOLD, unlocked.eye_pos_threshold);
    if let Some(size) = widget::Slider::new(unlocked.eye_pos_threshold as f32, 0f32, cmp::min(dim_img.0, dim_img.1) as f32)
        .w_h(275.0, 50.0)
        .mid_left_of(ids.canvas)
        .down_from(ids.eye_pos_xy, 15.0)
        .label(&eye_threshold_label)
        .set(ids.eye_pos_threshold_slider, ui) {
            unlocked.eye_pos_threshold = size as u32;
        };

    let eye_min_radius_label = format!("{} {}", EYE_MIN_RADIUS, unlocked.eye_min_radius);
    if let Some(size) = widget::Slider::new(unlocked.eye_min_radius, 0.0f32, cmp::min(dim_img.0, dim_img.1) as f32)
        .w_h(275.0, 50.0)
        .mid_left_of(ids.canvas)
        .down_from(ids.eye_pos_threshold_slider, 15.0)
        .label(&eye_min_radius_label)
        .set(ids.eye_min_radius_slider, ui) {
            unlocked.eye_min_radius = size;
        };

    let eye_max_radius_label = format!("{} {}", EYE_MAX_RADIUS, unlocked.eye_max_radius);
    if let Some(size) = widget::Slider::new(unlocked.eye_max_radius, 0.0f32, cmp::min(dim_img.0, dim_img.1) as f32)
        .w_h(275.0, 50.0)
        .mid_left_of(ids.canvas)
        .down_from(ids.eye_min_radius_slider, 15.0)
        .label(&eye_max_radius_label)
        .set(ids.eye_max_radius_slider, ui) {
            unlocked.eye_max_radius = size;
        };
}


pub fn draw_gui(app: Arc<Mutex<App>>, dim_img: (u32, u32)) {
    const WIDTH: u32 = 350;
    const HEIGHT: u32 = 735;
    let mut window: PistonWindow =
            WindowSettings::new("Config Settings", [WIDTH, HEIGHT])
                .opengl(OpenGL::V3_2) // If not working, try `OpenGL::V2_1`.
                .samples(4)
                .exit_on_esc(true)
                .vsync(true)
                .build()
                .unwrap();
    
    let mut ui = conrod::UiBuilder::new([WIDTH as f64, HEIGHT as f64])
        .theme(theme())
        .build();

    let assets = find_folder::Search::ParentsThenKids(3, 3)
        .for_folder("assets").unwrap();
    let font_path = assets.join("FiraSans-Regular.ttf");
    ui.fonts.insert_from_file(font_path).unwrap();

    let ids = Ids::new(ui.widget_id_generator());

    let image_map = conrod::image::Map::new();
    
    let mut text_vertex_data = Vec::new();
    let (mut glyph_cache, mut text_texture_cache) = {
        const SCALE_TOLERANCE: f32 = 0.1;
        const POSITION_TOLERANCE: f32 = 0.1;
        let cache = conrod::text::GlyphCache::new(WIDTH, HEIGHT, SCALE_TOLERANCE, POSITION_TOLERANCE);
        let buffer_len = WIDTH as usize * HEIGHT as usize;
        let init = vec![128; buffer_len];
        let settings = TextureSettings::new();
        let factory = &mut window.factory;
        let texture = G2dTexture::from_memory_alpha(factory, &init, WIDTH, HEIGHT, &settings).unwrap();
        (cache, texture)
    };

    while let Some(event) = window.next() {
        let size = window.size();
        let (win_w, win_h) = (size.width as conrod::Scalar, size.height as conrod::Scalar);
        if let Some(e) = conrod::backend::piston::event::convert(event.clone(), win_w, win_h) {
                ui.handle_event(e);
        }

        event.update(|_| {
            let mut ui = ui.set_widgets();
            let data = app.clone();
            gui(&mut ui, &ids, dim_img, data);
        });

         window.draw_2d(&event, |context, graphics| {
            if let Some(primitives) = ui.draw_if_changed() {
                let cache_queued_glyphs = |graphics: &mut G2d,
                                            cache: &mut G2dTexture,
                                            rect: conrod::text::rt::Rect<u32>,
                                            data: &[u8]| {
                    let offset = [rect.min.x, rect.min.y];
                    let size = [rect.width(), rect.height()];
                    let format = piston_window::texture::Format::Rgba8;
                    let encoder = &mut graphics.encoder;
                    text_vertex_data.clear();
                    text_vertex_data.extend(data.iter().flat_map(|&b| vec![255, 255, 255, b]));
                    UpdateTexture::update(cache, encoder, format, &text_vertex_data[..], offset, size)
                        .expect("failed to update texture")
                };

                fn texture_from_image<T>(img: &T) -> &T { img }
                conrod::backend::piston::draw::primitives(primitives,
                    context,
                    graphics,
                    &mut text_texture_cache,
                    &mut glyph_cache,
                    &image_map,
                    cache_queued_glyphs,
                    texture_from_image);
            }
        });
    }
}