#[derive(Debug, PartialEq)]
pub enum Viz3dCmd {
    On,
    Off,
    Reset,
}

pub fn parse_3d_command(input: &str) -> Option<Viz3dCmd> {
    match input.trim().to_ascii_lowercase().as_str() {
        ":3d" | ":3d on" => Some(Viz3dCmd::On),
        ":2d" | ":3d off" => Some(Viz3dCmd::Off),
        ":3d reset" => Some(Viz3dCmd::Reset),
        _ => None,
    }
}

pub fn is_3d_hotkey(ctrl: bool, code: &str) -> bool {
    ctrl && code == "Digit3"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands() {
        assert_eq!(parse_3d_command(":3d"), Some(Viz3dCmd::On));
        assert_eq!(parse_3d_command(":3d on"), Some(Viz3dCmd::On));
        assert_eq!(parse_3d_command(":2d"), Some(Viz3dCmd::Off));
        assert_eq!(parse_3d_command(":3d off"), Some(Viz3dCmd::Off));
        assert_eq!(parse_3d_command(":3d reset"), Some(Viz3dCmd::Reset));
        assert_eq!(parse_3d_command(":3D RESET"), Some(Viz3dCmd::Reset));
        assert_eq!(parse_3d_command(":help"), None);
    }
}
