use std::char;
use std::cmp::max;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::mem::swap;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CharacterSet {
    ranges: Vec<Range<u32>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum NfaState {
    Advance {
        chars: CharacterSet,
        state_id: u32,
        is_sep: bool,
        precedence: i32,
    },
    Split(u32, u32),
    Accept {
        variable_index: usize,
        precedence: i32,
    },
}

#[derive(PartialEq, Eq)]
pub struct Nfa {
    pub states: Vec<NfaState>,
}

#[derive(Debug)]
pub struct NfaCursor<'a> {
    pub(crate) state_ids: Vec<u32>,
    nfa: &'a Nfa,
}

#[derive(Debug, PartialEq, Eq)]
pub struct NfaTransition {
    pub characters: CharacterSet,
    pub is_separator: bool,
    pub precedence: i32,
    pub states: Vec<u32>,
}

impl Default for Nfa {
    fn default() -> Self {
        Self { states: Vec::new() }
    }
}

impl CharacterSet {
    pub fn empty() -> Self {
        CharacterSet { ranges: Vec::new() }
    }

    pub fn all() -> Self {
        CharacterSet {
            ranges: vec![0..u32::MAX],
        }
    }

    pub fn from_range(min: char, max: char) -> Self {
        let min = min as u32;
        let max = max as u32;
        CharacterSet {
            ranges: vec![min.min(max)..min.max(max)],
        }
    }

    pub fn from_char(c: char) -> Self {
        CharacterSet {
            ranges: vec![(c as u32)..(c as u32 + 1)],
        }
    }

    pub fn from_chars(chars: Vec<char>) -> Self {
        panic!("OH NO")
    }

    pub fn negate(self) -> CharacterSet {
        let mut previous_end = 0;
        let mut i = 0;
        while i < self.ranges.len() {
            let range = &mut self.ranges[i];
            range.end = range.start;
            range.start = previous_end;
            previous_end = range.end;
            if range.start == range.end {
                self.ranges.remove(i);
            } else {
                i += 1;
            }
        }
        if previous_end < u32::MAX {
            self.ranges.push(previous_end..u32::MAX);
        }
        self
    }

    pub fn add_char(self, c: char) -> Self {
        self.add_int_range(0, c as u32, c as u32 + 1);
        self
    }

    pub fn add_range(self, start: char, end: char) -> Self {
        self.add_int_range(0, start as u32, end as u32);
        self
    }

    pub fn add(self, other: &CharacterSet) -> Self {
        let mut index = 0;
        for range in &other.ranges {
            index = self.add_int_range(index, range.start as u32, range.end as u32);
        }
        self
    }

    fn add_int_range(self, mut i: usize, start: u32, end: u32) -> usize {
        while i < self.ranges.len() {
            let range = &mut self.ranges[i];
            if range.start > end {
                self.ranges.insert(i, start..end);
                return i;
            }
            if range.end >= start {
                range.end = range.end.max(end);
                range.start = range.start.min(start);
                return i;
            }
            i += 1;
        }
        self.ranges.push(start..end);
        i
    }

    fn remove_int_range(self, start: u32, end: u32) -> usize {
        let mut i = 0;
        while i < self.ranges.len() {
            let range = &mut self.ranges[i];
            if range.start > end {
                break;
            }
            if range.end > start {
                if range.end > end {
                    let new_range = range.start..start;
                    range.start = end;
                    self.ranges.insert(i, new_range);
                } else {
                    range.end = start;
                }
            }
            i += 1;
        }
        i
    }

    pub fn does_intersect(&self, other: &CharacterSet) -> bool {
        let mut left_ranges = self.ranges.iter();
        let mut right_ranges = other.ranges.iter();
        let mut left_range = left_ranges.next();
        let mut right_range = right_ranges.next();
        while let (Some(left), Some(right)) = (&left_range, &right_range) {
            if left.end <= right.start {
                left_range = left_ranges.next();
            } else if left.start >= right.end {
                right_range = right_ranges.next();
            } else {
                return true;
            }
        }
        false
    }

    pub fn remove_intersection(&mut self, other: &mut CharacterSet) -> CharacterSet {
        let mut intersection = Vec::new();
        let mut left_i = 0;
        let mut right_i = 0;
        while left_i < self.ranges.len() && right_i < other.ranges.len() {
            let left = &mut self.ranges[left_i];
            let right = &mut self.ranges[right_i];

            match left.start.cmp(&right.start) {
                Ordering::Less => {
                    // [ L ]
                    //     [ R ]
                    if left.end <= right.start {
                        left_i += 1;
                        continue;
                    }

                    match left.end.cmp(&right.end) {
                        // [ L ]
                        //   [ R ]
                        Ordering::Less => {
                            intersection.push(right.start..left.end);
                            left.end = right.start;
                            right.start = left.end;
                            left_i += 1;
                        }

                        // [  L  ]
                        //   [ R ]
                        Ordering::Equal => {
                            intersection.push(right.clone());
                            left.end = right.start;
                            other.ranges.remove(right_i);
                        }

                        // [   L   ]
                        //   [ R ]
                        Ordering::Greater => {
                            intersection.push(right.clone());
                            let new_range = left.start..right.start;
                            left.start = right.end;
                            self.ranges.insert(left_i, new_range);
                            left_i += 1;
                        }
                    }
                }
                Ordering::Equal => {
                    // [ L ]
                    // [  R  ]
                    if left.end < right.end {
                        intersection.push(left.start..left.end);
                        right.start = left.end;
                        self.ranges.remove(left_i);
                    }
                    // [ L ]
                    // [ R ]
                    else if left.end == right.end {
                        intersection.push(left.clone());
                        self.ranges.remove(left_i);
                        other.ranges.remove(right_i);
                    }
                    // [  L  ]
                    // [ R ]
                    else if left.end > right.end {
                        intersection.push(right.clone());
                        left.start = right.end;
                        other.ranges.remove(right_i);
                    }
                }
                Ordering::Greater => {
                    //     [ L ]
                    // [ R ]
                    if left.start >= right.end {
                        right_i += 1;
                        continue;
                    }

                    match left.end.cmp(&right.end) {
                        //   [ L ]
                        // [   R   ]
                        Ordering::Less => {
                            intersection.push(left.clone());
                            let new_range = right.start..left.start;
                            right.start = right.end;
                            other.ranges.insert(right_i, new_range);
                            right_i += 1;
                        }

                        //   [ L ]
                        // [  R  ]
                        Ordering::Equal => {
                            intersection.push(left.clone());
                            right.end = left.start;
                            self.ranges.remove(left_i);
                        }

                        //   [   L   ]
                        // [   R   ]
                        Ordering::Greater => {
                            intersection.push(left.start..right.end);
                            left.start = right.end;
                            right.end = left.start;
                            right_i += 1;
                        }
                    }
                }
            }
        }
        CharacterSet {
            ranges: intersection,
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.ranges.iter().flat_map(|r| r.clone())
    }

    pub fn chars<'a>(&'a self) -> impl Iterator<Item = char> + 'a {
        self.iter().filter_map(char::from_u32)
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub fn ranges<'a>(
        chars: &'a Vec<char>,
        ruled_out_characters: &'a HashSet<u32>,
    ) -> impl Iterator<Item = Range<char>> + 'a {
        let mut prev_range: Option<Range<char>> = None;
        chars
            .iter()
            .map(|c| (*c, false))
            .chain(Some(('\0', true)))
            .filter_map(move |(c, done)| {
                if done {
                    return prev_range.clone();
                }
                if ruled_out_characters.contains(&(c as u32)) {
                    return None;
                }
                if let Some(range) = prev_range.clone() {
                    let mut prev_range_successor = range.end as u32 + 1;
                    while prev_range_successor < c as u32 {
                        if !ruled_out_characters.contains(&prev_range_successor) {
                            prev_range = Some(c..c);
                            return Some(range);
                        }
                        prev_range_successor += 1;
                    }
                    prev_range = Some(range.start..c);
                    None
                } else {
                    prev_range = Some(c..c);
                    None
                }
            })
    }

    #[cfg(test)]
    pub fn contains(&self, c: char) -> bool {
        self.ranges.iter().any(|r| r.contains(&(c as u32)))
    }
}

impl Ord for CharacterSet {
    fn cmp(&self, other: &CharacterSet) -> Ordering {
        let has_max = self.ranges.last().map_or(false, |r| r.end == u32::MAX);
        let other_has_max = other.ranges.last().map_or(false, |r| r.end == u32::MAX);
        if has_max {
            if !other_has_max {
                return Ordering::Greater;
            }
        } else if other_has_max {
            return Ordering::Less;
        }
        self.chars().cmp(other.chars())
    }
}

impl PartialOrd for CharacterSet {
    fn partial_cmp(&self, other: &CharacterSet) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn add_chars(left: &mut Vec<char>, right: &Vec<char>) {
    for c in right {
        match left.binary_search(c) {
            Err(i) => left.insert(i, *c),
            _ => {}
        }
    }
}

fn remove_chars(left: &mut Vec<char>, right: &mut Vec<char>, mutate_right: bool) -> Vec<char> {
    let mut result = Vec::new();
    right.retain(|right_char| {
        if let Some(index) = left.iter().position(|left_char| *left_char == *right_char) {
            left.remove(index);
            result.push(*right_char);
            false || !mutate_right
        } else {
            true
        }
    });
    result
}

struct SetComparision {
    left_only: bool,
    common: bool,
    right_only: bool,
}

fn compare_chars(left: &Vec<char>, right: &Vec<char>) -> SetComparision {
    let mut result = SetComparision {
        left_only: false,
        common: false,
        right_only: false,
    };
    let mut left = left.iter().cloned();
    let mut right = right.iter().cloned();
    let mut i = left.next();
    let mut j = right.next();
    while let (Some(left_char), Some(right_char)) = (i, j) {
        if left_char < right_char {
            i = left.next();
            result.left_only = true;
        } else if left_char > right_char {
            j = right.next();
            result.right_only = true;
        } else {
            i = left.next();
            j = right.next();
            result.common = true;
        }
    }

    match (i, j) {
        (Some(_), _) => result.left_only = true,
        (_, Some(_)) => result.right_only = true,
        _ => {}
    }

    result
}

fn order_chars(chars: &Vec<char>, other_chars: &Vec<char>) -> Ordering {
    if chars.is_empty() {
        if other_chars.is_empty() {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    } else if other_chars.is_empty() {
        Ordering::Greater
    } else {
        let cmp = chars.len().cmp(&other_chars.len());
        if cmp != Ordering::Equal {
            return cmp;
        }
        for (c, other_c) in chars.iter().zip(other_chars.iter()) {
            let cmp = c.cmp(other_c);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }
}

impl Nfa {
    pub fn new() -> Self {
        Nfa { states: Vec::new() }
    }

    pub fn last_state_id(&self) -> u32 {
        self.states.len() as u32 - 1
    }
}

impl fmt::Debug for Nfa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Nfa {{ states: {{\n")?;
        for (i, state) in self.states.iter().enumerate() {
            write!(f, "  {}: {:?},\n", i, state)?;
        }
        write!(f, "}} }}")?;
        Ok(())
    }
}

impl<'a> NfaCursor<'a> {
    pub fn new(nfa: &'a Nfa, mut states: Vec<u32>) -> Self {
        let mut result = Self {
            nfa,
            state_ids: Vec::new(),
        };
        result.add_states(&mut states);
        result
    }

    pub fn reset(&mut self, mut states: Vec<u32>) {
        self.state_ids.clear();
        self.add_states(&mut states);
    }

    pub fn force_reset(&mut self, states: Vec<u32>) {
        self.state_ids = states
    }

    pub fn transition_chars(&self) -> impl Iterator<Item = (&CharacterSet, bool)> {
        self.raw_transitions().map(|t| (t.0, t.1))
    }

    pub fn transitions(&self) -> Vec<NfaTransition> {
        Self::group_transitions(self.raw_transitions())
    }

    fn raw_transitions(&self) -> impl Iterator<Item = (&CharacterSet, bool, i32, u32)> {
        self.state_ids.iter().filter_map(move |id| {
            if let NfaState::Advance {
                chars,
                state_id,
                precedence,
                is_sep,
            } = &self.nfa.states[*id as usize]
            {
                Some((chars, *is_sep, *precedence, *state_id))
            } else {
                None
            }
        })
    }

    fn group_transitions<'b>(
        iter: impl Iterator<Item = (&'b CharacterSet, bool, i32, u32)>,
    ) -> Vec<NfaTransition> {
        let mut result: Vec<NfaTransition> = Vec::new();
        for (chars, is_sep, prec, state) in iter {
            let mut chars = chars.clone();
            let mut i = 0;
            while i < result.len() && !chars.is_empty() {
                let intersection = result[i].characters.remove_intersection(&mut chars);
                if !intersection.is_empty() {
                    let mut intersection_states = result[i].states.clone();
                    match intersection_states.binary_search(&state) {
                        Err(j) => intersection_states.insert(j, state),
                        _ => {}
                    }
                    let intersection_transition = NfaTransition {
                        characters: intersection,
                        is_separator: result[i].is_separator && is_sep,
                        precedence: max(result[i].precedence, prec),
                        states: intersection_states,
                    };
                    if result[i].characters.is_empty() {
                        result[i] = intersection_transition;
                    } else {
                        result.insert(i, intersection_transition);
                        i += 1;
                    }
                }
                i += 1;
            }
            if !chars.is_empty() {
                result.push(NfaTransition {
                    characters: chars,
                    precedence: prec,
                    states: vec![state],
                    is_separator: is_sep,
                });
            }
        }
        result.sort_unstable_by(|a, b| a.characters.cmp(&b.characters));

        let mut i = 0;
        'i_loop: while i < result.len() {
            for j in 0..i {
                if result[j].states == result[i].states
                    && result[j].is_separator == result[i].is_separator
                    && result[j].precedence == result[i].precedence
                {
                    let mut characters = CharacterSet::empty();
                    swap(&mut characters, &mut result[j].characters);
                    result[j].characters = characters.add(&result[i].characters);
                    result.remove(i);
                    continue 'i_loop;
                }
            }
            i += 1;
        }

        result
    }

    pub fn completions(&self) -> impl Iterator<Item = (usize, i32)> + '_ {
        self.state_ids.iter().filter_map(move |state_id| {
            if let NfaState::Accept {
                variable_index,
                precedence,
            } = self.nfa.states[*state_id as usize]
            {
                Some((variable_index, precedence))
            } else {
                None
            }
        })
    }

    pub fn add_states(&mut self, new_state_ids: &mut Vec<u32>) {
        let mut i = 0;
        while i < new_state_ids.len() {
            let state_id = new_state_ids[i];
            let state = &self.nfa.states[state_id as usize];
            if let NfaState::Split(left, right) = state {
                let mut has_left = false;
                let mut has_right = false;
                for new_state_id in new_state_ids.iter() {
                    if *new_state_id == *left {
                        has_left = true;
                    }
                    if *new_state_id == *right {
                        has_right = true;
                    }
                }
                if !has_left {
                    new_state_ids.push(*left);
                }
                if !has_right {
                    new_state_ids.push(*right);
                }
            } else if let Err(i) = self.state_ids.binary_search(&state_id) {
                self.state_ids.insert(i, state_id);
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_transitions() {
        let table = [
            // overlapping character classes
            (
                vec![
                    (CharacterSet::empty().add_range('a', 'f'), false, 0, 1),
                    (CharacterSet::empty().add_range('d', 'i'), false, 1, 2),
                ],
                vec![
                    NfaTransition {
                        characters: CharacterSet::empty().add_range('a', 'c'),
                        is_separator: false,
                        precedence: 0,
                        states: vec![1],
                    },
                    NfaTransition {
                        characters: CharacterSet::empty().add_range('d', 'f'),
                        is_separator: false,
                        precedence: 1,
                        states: vec![1, 2],
                    },
                    NfaTransition {
                        characters: CharacterSet::empty().add_range('g', 'i'),
                        is_separator: false,
                        precedence: 1,
                        states: vec![2],
                    },
                ],
            ),
            // large character class followed by many individual characters
            (
                vec![
                    (CharacterSet::empty().add_range('a', 'z'), false, 0, 1),
                    (CharacterSet::empty().add_char('d'), false, 0, 2),
                    (CharacterSet::empty().add_char('i'), false, 0, 3),
                    (CharacterSet::empty().add_char('f'), false, 0, 4),
                ],
                vec![
                    NfaTransition {
                        characters: CharacterSet::empty().add_char('d'),
                        is_separator: false,
                        precedence: 0,
                        states: vec![1, 2],
                    },
                    NfaTransition {
                        characters: CharacterSet::empty().add_char('f'),
                        is_separator: false,
                        precedence: 0,
                        states: vec![1, 4],
                    },
                    NfaTransition {
                        characters: CharacterSet::empty().add_char('i'),
                        is_separator: false,
                        precedence: 0,
                        states: vec![1, 3],
                    },
                    NfaTransition {
                        characters: CharacterSet::empty()
                            .add_range('a', 'c')
                            .add_char('e')
                            .add_range('g', 'h')
                            .add_range('j', 'z'),
                        is_separator: false,
                        precedence: 0,
                        states: vec![1],
                    },
                ],
            ),
            // negated character class followed by an individual character
            (
                vec![
                    (CharacterSet::empty().add_char('0'), false, 0, 1),
                    (CharacterSet::empty().add_char('b'), false, 0, 2),
                    (
                        CharacterSet::empty().add_range('a', 'f').negate(),
                        false,
                        0,
                        3,
                    ),
                    (CharacterSet::empty().add_char('c'), false, 0, 4),
                ],
                vec![
                    NfaTransition {
                        characters: CharacterSet::empty().add_char('0'),
                        precedence: 0,
                        states: vec![1, 3],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::empty().add_char('b'),
                        precedence: 0,
                        states: vec![2],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::empty().add_char('c'),
                        precedence: 0,
                        states: vec![4],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::empty()
                            .add_range('a', 'f')
                            .add_char('0')
                            .negate(),
                        precedence: 0,
                        states: vec![3],
                        is_separator: false,
                    },
                ],
            ),
            // multiple negated character classes
            (
                vec![
                    (CharacterSet::from_char('a'), false, 0, 1),
                    (CharacterSet::from_range('a', 'c').negate(), false, 0, 2),
                    (CharacterSet::from_char('g'), false, 0, 6),
                    (CharacterSet::from_range('d', 'f').negate(), false, 0, 3),
                    (CharacterSet::from_range('g', 'i').negate(), false, 0, 4),
                    (CharacterSet::from_char('g'), false, 0, 5),
                ],
                vec![
                    NfaTransition {
                        characters: CharacterSet::from_char('a'),
                        precedence: 0,
                        states: vec![1, 3, 4],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::from_char('g'),
                        precedence: 0,
                        states: vec![2, 3, 5, 6],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::from_range('b', 'c'),
                        precedence: 0,
                        states: vec![3, 4],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::from_range('h', 'i'),
                        precedence: 0,
                        states: vec![2, 3],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::from_range('d', 'f'),
                        precedence: 0,
                        states: vec![2, 4],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::from_range('a', 'i').negate(),
                        precedence: 0,
                        states: vec![2, 3, 4],
                        is_separator: false,
                    },
                ],
            ),
            // disjoint characters with same state
            (
                vec![
                    (CharacterSet::from_char('a'), false, 0, 1),
                    (CharacterSet::from_char('b'), false, 0, 2),
                    (CharacterSet::from_char('c'), false, 0, 1),
                    (CharacterSet::from_char('d'), false, 0, 1),
                    (CharacterSet::from_char('e'), false, 0, 2),
                ],
                vec![
                    NfaTransition {
                        characters: CharacterSet::from_chars(vec!['a', 'c', 'd']),
                        precedence: 0,
                        states: vec![1],
                        is_separator: false,
                    },
                    NfaTransition {
                        characters: CharacterSet::from_chars(vec!['b', 'e']),
                        precedence: 0,
                        states: vec![2],
                        is_separator: false,
                    },
                ],
            ),
        ];

        for row in table.iter() {
            assert_eq!(
                NfaCursor::group_transitions(
                    row.0
                        .iter()
                        .map(|(chars, is_sep, prec, state)| (chars, *is_sep, *prec, *state))
                ),
                row.1
            );
        }
    }

    #[test]
    fn test_character_set_remove_intersection() {
        // A whitelist and an overlapping whitelist.
        // Both sets contain 'c', 'd', and 'f'
        let mut a = CharacterSet::from_range('a', 'f');
        let mut b = CharacterSet::from_range('c', 'h');
        assert_eq!(
            a.remove_intersection(&mut b),
            CharacterSet::from_range('c', 'f')
        );
        assert_eq!(a, CharacterSet::from_range('a', 'b'));
        assert_eq!(b, CharacterSet::from_range('g', 'h'));

        let mut a = CharacterSet::from_range('a', 'f');
        let mut b = CharacterSet::from_range('c', 'h');
        assert_eq!(
            b.remove_intersection(&mut a),
            CharacterSet::from_range('c', 'f')
        );
        assert_eq!(a, CharacterSet::from_range('a', 'b'));
        assert_eq!(b, CharacterSet::from_range('g', 'h'));

        // A whitelist and a larger whitelist.
        let mut a = CharacterSet::from_char('c');
        let mut b = CharacterSet::from_range('a', 'e');
        assert_eq!(a.remove_intersection(&mut b), CharacterSet::from_char('c'));
        assert_eq!(a, CharacterSet::empty());
        assert_eq!(
            b,
            CharacterSet::empty()
                .add_range('a', 'b')
                .add_range('d', 'e')
        );

        let mut a = CharacterSet::empty().add_char('c');
        let mut b = CharacterSet::empty().add_range('a', 'e');
        assert_eq!(
            b.remove_intersection(&mut a),
            CharacterSet::empty().add_char('c')
        );
        assert_eq!(a, CharacterSet::empty());
        assert_eq!(
            b,
            CharacterSet::empty()
                .add_range('a', 'b')
                .add_range('d', 'e')
        );

        // An inclusion and an intersecting exclusion.
        // Both sets contain 'e', 'f', and 'm'
        let mut a = CharacterSet::empty()
            .add_range('c', 'h')
            .add_range('k', 'm');
        let mut b = CharacterSet::empty()
            .add_range('a', 'd')
            .add_range('g', 'l')
            .negate();
        assert_eq!(
            a.remove_intersection(&mut b),
            CharacterSet::from_chars(vec!['e', 'f', 'm'])
        );
        assert_eq!(
            a,
            CharacterSet::from_chars(vec!['c', 'd', 'g', 'h', 'k', 'l'])
        );
        assert_eq!(b, CharacterSet::empty().add_range('a', 'm').negate());

        let mut a = CharacterSet::empty()
            .add_range('c', 'h')
            .add_range('k', 'm');
        let mut b = CharacterSet::empty()
            .add_range('a', 'd')
            .add_range('g', 'l')
            .negate();
        assert_eq!(
            b.remove_intersection(&mut a),
            CharacterSet::from_chars(vec!['e', 'f', 'm'])
        );
        assert_eq!(
            a,
            CharacterSet::from_chars(vec!['c', 'd', 'g', 'h', 'k', 'l'])
        );
        assert_eq!(b, CharacterSet::empty().add_range('a', 'm').negate());

        // An exclusion and an overlapping inclusion.
        // Both sets exclude 'c', 'd', and 'e'
        let mut a = CharacterSet::empty().add_range('a', 'e').negate();
        let mut b = CharacterSet::empty().add_range('c', 'h').negate();
        assert_eq!(
            a.remove_intersection(&mut b),
            CharacterSet::empty().add_range('a', 'h').negate(),
        );
        assert_eq!(a, CharacterSet::from_range('f', 'h'));
        assert_eq!(b, CharacterSet::from_range('a', 'b'));

        // An exclusion and a larger exclusion.
        let mut a = CharacterSet::empty().add_range('b', 'c').negate();
        let mut b = CharacterSet::empty().add_range('a', 'd').negate();
        assert_eq!(
            a.remove_intersection(&mut b),
            CharacterSet::empty().add_range('a', 'd').negate(),
        );
        assert_eq!(a, CharacterSet::empty().add_char('a').add_char('d'));
        assert_eq!(b, CharacterSet::empty());
    }

    #[test]
    fn test_character_set_does_intersect() {
        let (a, b) = (CharacterSet::empty(), CharacterSet::empty());
        assert!(!a.does_intersect(&b));
        assert!(!b.does_intersect(&a));

        let (a, b) = (
            CharacterSet::empty().add_char('a'),
            CharacterSet::empty().add_char('a'),
        );
        assert!(a.does_intersect(&b));
        assert!(b.does_intersect(&a));

        let (a, b) = (
            CharacterSet::empty().add_char('b'),
            CharacterSet::empty().add_char('a').add_char('c'),
        );
        assert!(!a.does_intersect(&b));
        assert!(!b.does_intersect(&a));

        let (a, b) = (
            CharacterSet::from_char('b'),
            CharacterSet::from_range('a', 'c').negate(),
        );
        assert!(!a.does_intersect(&b));
        assert!(!b.does_intersect(&a));

        let (a, b) = (
            CharacterSet::from_char('b'),
            CharacterSet::from_range('a', 'c').negate(),
        );
        assert!(a.does_intersect(&b));
        assert!(b.does_intersect(&a));

        let (a, b) = (
            CharacterSet::from_char('a').negate(),
            CharacterSet::from_char('a').negate(),
        );
        assert!(a.does_intersect(&b));
        assert!(b.does_intersect(&a));

        let (a, b) = (
            CharacterSet::from_char('c'),
            CharacterSet::from_char('a').negate(),
        );
        assert!(a.does_intersect(&b));
        assert!(b.does_intersect(&a));
    }

    #[test]
    fn test_character_set_get_ranges() {
        struct Row {
            chars: Vec<char>,
            ruled_out_chars: Vec<char>,
            expected_ranges: Vec<Range<char>>,
        }

        let table = [
            Row {
                chars: vec!['a'],
                ruled_out_chars: vec![],
                expected_ranges: vec!['a'..'a'],
            },
            Row {
                chars: vec!['a', 'b', 'c', 'e', 'z'],
                ruled_out_chars: vec![],
                expected_ranges: vec!['a'..'c', 'e'..'e', 'z'..'z'],
            },
            Row {
                chars: vec!['a', 'b', 'c', 'e', 'h', 'z'],
                ruled_out_chars: vec!['d', 'f', 'g'],
                expected_ranges: vec!['a'..'h', 'z'..'z'],
            },
        ];

        for Row {
            chars,
            ruled_out_chars,
            expected_ranges,
        } in table.iter()
        {
            let ruled_out_chars = ruled_out_chars
                .into_iter()
                .map(|c: &char| *c as u32)
                .collect();
            let ranges = CharacterSet::ranges(chars, &ruled_out_chars).collect::<Vec<_>>();
            assert_eq!(ranges, *expected_ranges);
        }
    }
}
