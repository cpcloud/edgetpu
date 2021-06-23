use crate::error::Error;
use ordered_float::NotNan;
use std::ops::{Add, Sub};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub(super) struct Point {
    x: f32,
    y: f32,
}

impl Point {
    pub(super) fn new(x: f32, y: f32) -> Result<Self, Error> {
        Ok(Self {
            x: NotNan::new(x)
                .map_err(|e| Error::ConstructNotNan(e, x))?
                .into_inner(),
            y: NotNan::new(y)
                .map_err(|e| Error::ConstructNotNan(e, y))?
                .into_inner(),
        })
    }

    pub(super) fn squared_distance(self, other: Self) -> f32 {
        let delta = other - self;
        delta.dot(delta)
    }

    #[inline]
    pub(super) fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub(super) fn x(self) -> f32 {
        self.x
    }

    #[inline]
    pub(super) fn y(self) -> f32 {
        self.y
    }

    #[inline]
    pub(super) fn x_mut(&mut self) -> &mut f32 {
        &mut self.x
    }

    #[inline]
    pub(super) fn y_mut(&mut self) -> &mut f32 {
        &mut self.y
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Point;

    #[test]
    fn xy_points() {
        let a = Point::new(0.5, 0.5).unwrap();
        let b = Point::new(1.0, 1.0).unwrap();
        assert_eq!(a.squared_distance(b), 0.5);
    }
}
