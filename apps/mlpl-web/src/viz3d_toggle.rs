pub fn parse_3d_command(input: &str) -> Option<Option<bool>> {
    match input.trim() {
        ":3d" => Some(None),
        ":3d on" => Some(Some(true)),
        ":3d off" => Some(Some(false)),
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
    fn parse_3d_commands() {
        assert_eq!(parse_3d_command(":3d"), Some(None));
        assert_eq!(parse_3d_command(":3d on"), Some(Some(true)));
        assert_eq!(parse_3d_command(":3d off"), Some(Some(false)));
        assert_eq!(parse_3d_command(":help"), None);
    }
}
