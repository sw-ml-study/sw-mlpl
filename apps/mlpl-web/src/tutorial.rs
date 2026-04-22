use yew::prelude::*;

pub use crate::lessons::{LESSONS, Lesson};

pub fn toggle_tutorial(lesson: UseStateHandle<Option<usize>>) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        if lesson.is_some() {
            lesson.set(None);
        } else {
            lesson.set(Some(0));
        }
    })
}

pub fn step_lesson(
    lesson: UseStateHandle<Option<usize>>,
    delta: i32,
) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        if let Some(cur) = *lesson {
            let next = i32::try_from(cur).unwrap_or(0) + delta;
            if next >= 0 {
                let next_usize = next as usize;
                if next_usize < LESSONS.len() {
                    lesson.set(Some(next_usize));
                }
            }
        }
    })
}

pub fn run_example(
    on_submit: Callback<String>,
    input_value: UseStateHandle<String>,
) -> Callback<String> {
    Callback::from(move |line: String| {
        input_value.set(line.clone());
        on_submit.emit(line);
    })
}
