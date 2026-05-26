pub fn parse_3d_command(input: &str) -> Option<bool> {
    let lower = input.trim().to_ascii_lowercase();
    match lower.as_str() {
        ":3d" | ":3d on" => Some(true),
        ":2d" | ":3d off" => Some(false),
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
    fn idempotent_on_off() {
        assert_eq!(parse_3d_command(":3d"), Some(true));
        assert_eq!(parse_3d_command(":3d on"), Some(true));
        assert_eq!(parse_3d_command(":2d"), Some(false));
        assert_eq!(parse_3d_command(":3d off"), Some(false));
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(parse_3d_command(":3D"), Some(true));
        assert_eq!(parse_3d_command(":3D ON"), Some(true));
        assert_eq!(parse_3d_command(":2D"), Some(false));
        assert_eq!(parse_3d_command(":3D OFF"), Some(false));
    }

    #[test]
    fn non_commands() {
        assert_eq!(parse_3d_command(":help"), None);
        assert_eq!(parse_3d_command("3d"), None);
    }
}
